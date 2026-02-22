from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .auto_review import build_auto_review, write_auto_review
from .gpu_runtime import dataframe_backend_state, maybe_enable_cudf_pandas
from .gpu_monitor import GPUSampler

maybe_enable_cudf_pandas()

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import config_hash, load_config
from .data_store import update_data
from .evaluation import evaluate_predictions
from .features import build_feature_store
from .fmp_client import FMPClient
from .labels import build_future_return_labels
from .logging_utils import capture_env_snapshot, get_default_secrets_to_mask, setup_logging
from .paths import ProjectPaths
from .reporting import render_report_md, write_report
from .splits import apply_purge_embargo, make_walkforward_splits, run_leakage_gate
from .state import (
    STAGES_SUCCESS_ORDER,
    acquire_lock,
    append_state_error,
    last_success_stage,
    load_last_state,
    load_run_state,
    normalize_state,
    release_lock,
    save_last_state,
)
from .training import train_models
from .validation import validate_raw_prices

logger = logging.getLogger(__name__)


class GateFailed(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_code_hash(src_dir: Path) -> str:
    files = sorted((src_dir / "gbdt_agent").rglob("*.py"), key=lambda p: str(p))
    h = hashlib.sha1()
    for f in files:
        h.update(str(f.relative_to(src_dir)).encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def make_run_id(conf_hash_val: str, code_hash: str) -> str:
    return f"{_utc_ts()}_{conf_hash_val[:8]}_{code_hash[:8]}"


def _safe_stage_index(stage: str) -> int:
    return STAGES_SUCCESS_ORDER.index(stage)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def _summarize_error(exc: BaseException) -> Dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:600],
        "traceback": traceback.format_exc(limit=50),
    }


def _apply_feature_quality_filter(
    *,
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    train_dates: Sequence,
    min_finite_ratio: float,
    min_nonzero_ratio: float,
    min_std: float,
    nonzero_eps: float,
) -> tuple[list[str], Dict[str, Any]]:
    cols = [str(c) for c in feature_cols]
    if not cols:
        return [], {"enabled": False, "kept_count": 0, "dropped_count": 0}

    if min_finite_ratio <= 0.0 and min_nonzero_ratio <= 0.0 and min_std <= 0.0:
        return cols, {"enabled": False, "kept_count": len(cols), "dropped_count": 0}

    train = merged[merged["decision_date"].isin(train_dates)].copy()
    kept: list[str] = []
    dropped: list[Dict[str, Any]] = []

    for c in cols:
        s = pd.to_numeric(train[c], errors="coerce")
        finite_mask = np.isfinite(s.to_numpy(dtype=float, copy=False))
        finite_n = int(finite_mask.sum())
        finite_ratio = float(finite_n / max(1, len(train)))
        if finite_n <= 0:
            dropped.append(
                {
                    "feature": c,
                    "reason": "no_finite_values",
                    "finite_count": 0,
                    "finite_ratio": 0.0,
                    "nonzero_ratio": None,
                    "std": None,
                }
            )
            continue
        if finite_ratio < float(min_finite_ratio):
            dropped.append(
                {
                    "feature": c,
                    "reason": "low_finite_ratio",
                    "finite_count": finite_n,
                    "finite_ratio": finite_ratio,
                    "nonzero_ratio": None,
                    "std": None,
                }
            )
            continue

        x = s.to_numpy(dtype=float, copy=False)[finite_mask]
        nonzero_ratio = float(np.mean(np.abs(x) > float(nonzero_eps)))
        std_val = float(np.std(x))

        if nonzero_ratio < float(min_nonzero_ratio):
            dropped.append(
                {
                    "feature": c,
                    "reason": "low_nonzero_ratio",
                    "finite_count": finite_n,
                    "finite_ratio": finite_ratio,
                    "nonzero_ratio": nonzero_ratio,
                    "std": std_val,
                }
            )
            continue
        if std_val < float(min_std):
            dropped.append(
                {
                    "feature": c,
                    "reason": "low_std",
                    "finite_count": finite_n,
                    "finite_ratio": finite_ratio,
                    "nonzero_ratio": nonzero_ratio,
                    "std": std_val,
                }
            )
            continue
        kept.append(c)

    summary: Dict[str, Any] = {
        "enabled": True,
        "min_finite_ratio": float(min_finite_ratio),
        "min_nonzero_ratio": float(min_nonzero_ratio),
        "min_std": float(min_std),
        "nonzero_eps": float(nonzero_eps),
        "before_count": int(len(cols)),
        "kept_count": int(len(kept)),
        "dropped_count": int(len(dropped)),
        "dropped_examples": dropped[:25],
    }
    return kept, summary


def _stage_artifacts_exist(
    *,
    stage: str,
    paths: ProjectPaths,
    run_dir: Path,
    state: Dict[str, Any],
    metrics: Dict[str, Any],
) -> bool:
    dataset_id = state.get("dataset_id") or metrics.get("dataset_id")
    feature_store_id = state.get("feature_store_id") or metrics.get("feature_store_id")

    if stage == "stage_00_env_ready":
        env = state.get("env_snapshot_path") or metrics.get("env_snapshot_path")
        return bool(env and Path(env).exists())

    if stage == "stage_10_data_ready":
        if not dataset_id:
            return False
        raw = paths.raw_dir / str(dataset_id)
        return all((raw / x).exists() for x in ["prices.parquet", "universe_membership.parquet", "dataset_spec.json"])

    if stage == "stage_20_validation_passed":
        if not dataset_id:
            return False
        proc = paths.processed_dir / str(dataset_id)
        return (proc / "prices_validated.parquet").exists() and (proc / "universe_membership_validated.parquet").exists()

    if stage == "stage_30_features_ready":
        if not dataset_id or not feature_store_id:
            return False
        proc = paths.processed_dir / str(dataset_id)
        fs = paths.feature_store_dir / str(feature_store_id)
        return (proc / "labels.parquet").exists() and (proc / "merged.parquet").exists() and (fs / "features.parquet").exists()

    if stage == "stage_40_split_leakcheck_passed":
        return (run_dir / "leakage.json").exists()

    if stage == "stage_50_models_trained":
        ckpts = state.get("model_ckpt_paths") or metrics.get("model_ckpt_paths") or {}
        if not isinstance(ckpts, dict) or "gbdt" not in ckpts:
            return False
        return Path(str(ckpts["gbdt"])).exists()

    if stage == "stage_60_predictions_ready":
        return (run_dir / "predictions.parquet").exists()

    if stage == "stage_70_backtest_ready":
        return (run_dir / "backtest.parquet").exists() and (run_dir / "backtest_positions.parquet").exists()

    if stage == "stage_80_report_ready":
        return (run_dir / "metrics.json").exists() and (run_dir / "report.md").exists()

    return False


def _write_artifact_manifest(paths: ProjectPaths, run_dir: Path, run_id: str, stage: str) -> Path:
    files = sorted([str(p.relative_to(paths.project_dir)) for p in run_dir.rglob("*") if p.is_file()])
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "generated_at": _utc_iso(),
        "github_lightweight": [
            "conf/",
            "src/",
            "notebooks/",
            f"artifacts/runs/{run_id}/metrics.json",
            f"artifacts/runs/{run_id}/report.md",
            "reports/",
        ],
        "drive_heavy": [
            "data/raw/",
            "data/cache_http/",
            "artifacts/runs/",
            "state/",
            "logs/",
        ],
        "run_files": files,
    }
    out = run_dir / "artifact_manifest.json"
    _write_json(out, manifest)
    return out


def run_pipeline(
    *,
    project_dir: Path,
    conf_path: Path,
    resume: bool,
    resume_run_id: Optional[str] = None,
    allow_resume_conf_mismatch: bool = False,
    force_stage: Optional[str] = None,
    force_unlock: bool = False,
    stop_after_stage: Optional[str] = None,
) -> str:
    paths = ProjectPaths.from_project_dir(project_dir)
    paths.ensure_base_dirs()

    cfg_file = conf_path if conf_path.is_absolute() else (project_dir / conf_path)
    cfg = load_config(cfg_file)
    cfg.setdefault("project", {})["project_dir"] = str(project_dir)

    conf_hash_val = config_hash(cfg)
    code_hash = compute_code_hash(paths.src_dir)

    last_state: Optional[Dict[str, Any]] = None
    if resume:
        if resume_run_id:
            loaded = load_run_state(paths.state_dir, str(resume_run_id))
            if not loaded:
                raise RuntimeError(f"Requested resume run_id not found in state/runs: {resume_run_id}")
            last_state = normalize_state(loaded)
        else:
            loaded = load_last_state(paths.state_dir)
            last_state = normalize_state(loaded) if loaded else None

    reuse_run = False
    if last_state and last_state.get("run_id"):
        conf_matches = (last_state.get("conf_hash") == conf_hash_val)
        if conf_matches:
            reuse_run = True
        elif resume_run_id and allow_resume_conf_mismatch:
            reuse_run = True
            logger.warning(
                "resume_conf_hash_mismatch_allowed run_id=%s state_conf=%s current_conf=%s",
                last_state.get("run_id"),
                last_state.get("conf_hash"),
                conf_hash_val,
            )
        elif resume_run_id:
            raise RuntimeError(
                "Resume target conf_hash mismatch. Re-run with --allow-conf-mismatch-resume to force continuation."
            )

    run_id = str(last_state.get("run_id")) if reuse_run else make_run_id(conf_hash_val, code_hash)

    run_dir = paths.run_dir(run_id)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "model_ckpt").mkdir(parents=True, exist_ok=True)

    setup_logging(paths.run_log_path(run_id), level=str(cfg.get("run", {}).get("log_level", "INFO")), secrets=get_default_secrets_to_mask())
    acquire_lock(paths.state_dir, run_id, force=force_unlock)

    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "conf_hash": conf_hash_val,
        "code_hash": code_hash,
        "status": "running",
        "errors": [],
        "historical_errors": [],
        "runtime": {"dataframe_backend": dataframe_backend_state()},
    }

    metrics_path = run_dir / "metrics.json"
    if reuse_run and metrics_path.exists():
        old = json.loads(metrics_path.read_text())
        if isinstance(old, dict):
            old.update({"run_id": run_id, "conf_hash": conf_hash_val, "code_hash": code_hash, "status": "running"})
            old_errors = old.get("errors") if isinstance(old.get("errors"), list) else []
            old_hist = old.get("historical_errors") if isinstance(old.get("historical_errors"), list) else []
            old["historical_errors"] = (old_hist + old_errors)[-200:]
            # Keep only active errors for the current execution attempt.
            old["errors"] = []
            metrics = old
    runtime_info = metrics.get("runtime")
    if not isinstance(runtime_info, dict):
        runtime_info = {}
        metrics["runtime"] = runtime_info
    runtime_info["dataframe_backend"] = dataframe_backend_state()

    state_payload: Dict[str, Any] = {
        "run_id": run_id,
        "stage": "stage_00_env_ready",
        "conf_hash": conf_hash_val,
        "dataset_id": None,
        "feature_store_id": None,
        "train_window": None,
        "val_window": None,
        "test_window": None,
        "model_ckpt_paths": {},
        "last_data_date": None,
        "errors": [],
        "code_hash": code_hash,
        "env_snapshot_path": None,
        "artifacts_dir": str(run_dir),
        "updated_at": None,
    }
    if reuse_run and last_state:
        for k in [
            "dataset_id",
            "feature_store_id",
            "train_window",
            "val_window",
            "test_window",
            "model_ckpt_paths",
            "last_data_date",
            "errors",
            "env_snapshot_path",
            "artifacts_dir",
        ]:
            state_payload[k] = last_state.get(k)
    state_payload = normalize_state(state_payload)

    horizon = int((cfg.get("labels") or {}).get("horizon_trading_days", 20))
    target_col = str((cfg.get("labels") or {}).get("target_col", f"future_return_{horizon}d"))
    run_cfg = cfg.get("run") or {}
    drive_cfg = cfg.get("drive") if isinstance(cfg.get("drive"), dict) else {}
    checkpoint_sync_cfg = run_cfg.get("checkpoint_sync") if isinstance(run_cfg.get("checkpoint_sync"), dict) else {}
    checkpoint_sync_use_drive_cfg = bool(checkpoint_sync_cfg.get("use_drive_config_path", False))
    checkpoint_drive_raw = str(
        run_cfg.get("checkpoint_drive_path")
        or os.environ.get("GBDT_CHECKPOINT_DRIVE_PATH", "")
        or (
            (drive_cfg.get("drive_path") if bool(drive_cfg.get("enabled", False)) else "")
            if checkpoint_sync_use_drive_cfg
            else ""
        )
    ).strip()
    checkpoint_drive_path = Path(checkpoint_drive_raw) if checkpoint_drive_raw else None
    if checkpoint_drive_path:
        metrics["checkpoint_drive_path"] = str(checkpoint_drive_path)
    checkpoint_sync_default_mode = str(checkpoint_sync_cfg.get("mode", "quick")).strip().lower()
    if checkpoint_sync_default_mode not in {"quick", "full"}:
        checkpoint_sync_default_mode = "quick"
    checkpoint_sync_min_interval = max(0.0, float(checkpoint_sync_cfg.get("min_interval_seconds", 0.0)))
    checkpoint_sync_full_stages = set(
        str(x).strip()
        for x in (checkpoint_sync_cfg.get("full_stages") or ["stage_80_report_ready"])
        if str(x).strip()
    )
    last_checkpoint_sync_at = 0.0

    auto_review_cfg = run_cfg.get("auto_review") if isinstance(run_cfg.get("auto_review"), dict) else {}
    auto_review_enabled = bool(auto_review_cfg.get("enabled", True))
    auto_review_strict = bool(auto_review_cfg.get("strict", False))
    auto_review_stages = set(
        str(x).strip()
        for x in (
            auto_review_cfg.get("stages")
            or ["stage_60_predictions_ready", "stage_70_backtest_ready", "stage_80_report_ready"]
        )
        if str(x).strip()
    )

    runtime_cfg = (cfg.get("runtime") or {}).get("gpu_sampling") or {}
    gpu_sampler = GPUSampler(
        enabled=bool(runtime_cfg.get("enabled", True)),
        sample_interval_sec=float(runtime_cfg.get("sample_interval_sec", 1.0)),
        active_util_threshold=float(runtime_cfg.get("active_util_threshold", 5.0)),
        active_mem_threshold_mib=float(runtime_cfg.get("active_mem_threshold_mib", 128.0)),
    )
    metrics["runtime"]["gpu_sampling"] = gpu_sampler.start(log_path=run_dir / "logs" / "gpu_samples.csv")
    gpu_sampling_finalized = False

    def _finalize_gpu_sampling() -> None:
        nonlocal gpu_sampling_finalized
        if gpu_sampling_finalized:
            return
        gpu_sampling_finalized = True
        try:
            metrics.setdefault("runtime", {})["gpu_sampling"] = gpu_sampler.stop()
        except Exception as exc:
            logger.warning("gpu_sampling_finalize_failed err=%s", exc)

    def _checkpoint_sync(stage: str) -> None:
        nonlocal last_checkpoint_sync_at
        if checkpoint_drive_path is None:
            return
        log_path = run_dir / "logs" / "checkpoint_sync.log"
        mode = "full" if stage in checkpoint_sync_full_stages else checkpoint_sync_default_mode
        now_ts = time.time()
        if mode != "full" and checkpoint_sync_min_interval > 0:
            elapsed = now_ts - last_checkpoint_sync_at
            if elapsed < checkpoint_sync_min_interval:
                rec = {
                    "ts": _utc_iso(),
                    "stage": stage,
                    "drive_path": str(checkpoint_drive_path),
                    "mode": mode,
                    "skipped": True,
                    "reason": "min_interval_not_elapsed",
                    "elapsed_seconds": round(elapsed, 3),
                    "min_interval_seconds": checkpoint_sync_min_interval,
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                return
        try:
            from .colab import sync_runtime_to_drive

            stats = sync_runtime_to_drive(
                local_root=project_dir,
                drive_path=checkpoint_drive_path,
                mode=mode,
                run_id=run_id,
            )
            last_checkpoint_sync_at = time.time()
            rec = {
                "ts": _utc_iso(),
                "stage": stage,
                "drive_path": str(checkpoint_drive_path),
                "mode": mode,
                "copied_files": int(stats.get("copied_files", 0)),
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        except Exception as exc:  # pragma: no cover
            rec = {
                "ts": _utc_iso(),
                "stage": stage,
                "drive_path": str(checkpoint_drive_path),
                "mode": mode,
                "error": f"{type(exc).__name__}: {exc}",
            }
            logger.warning("checkpoint_sync_failed stage=%s err=%s", stage, rec["error"])
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    def _run_auto_review(stage: str) -> None:
        if not auto_review_enabled or stage not in auto_review_stages:
            return
        log_path = run_dir / "logs" / "auto_review.log"
        try:
            payload = build_auto_review(run_id=run_id, stage=stage, run_dir=run_dir, metrics=metrics)
            out_path = write_auto_review(run_dir=run_dir, payload=payload)
            metrics["auto_review"] = payload
            rec = {
                "ts": _utc_iso(),
                "stage": stage,
                "status": str(payload.get("status", "unknown")),
                "warnings": len(payload.get("warnings") or []),
                "criticals": len(payload.get("criticals") or []),
                "path": str(out_path),
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            if auto_review_strict and str(payload.get("status")) == "fail":
                raise RuntimeError(f"auto_review_failed stage={stage} path={out_path}")
        except Exception as exc:  # pragma: no cover
            rec = {
                "ts": _utc_iso(),
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            if auto_review_strict:
                raise

    def _save_state(stage: str) -> None:
        state_payload["stage"] = stage
        save_last_state(paths.state_dir, state_payload)
        _write_artifact_manifest(paths, run_dir, run_id, stage)
        _checkpoint_sync(stage)

    def _write_metrics() -> None:
        _write_json(metrics_path, metrics)

    def _write_report(status: str, errors: Optional[Any] = None) -> None:
        report_errors = errors if errors is not None else metrics.get("errors")
        report = render_report_md(
            cfg=cfg,
            run_id=run_id,
            conf_hash=conf_hash_val,
            dataset_id=metrics.get("dataset_id"),
            feature_store_id=metrics.get("feature_store_id"),
            code_hash=code_hash,
            universe_info=metrics.get("universe"),
            split_info=metrics.get("split_info"),
            validation=metrics.get("validation"),
            leakage=metrics.get("leakage"),
            training_info=metrics.get("training_info"),
            model_metrics=metrics.get("model_metrics"),
            chosen_model=(metrics.get("backtest") or {}).get("chosen_model"),
            backtest_summary=(metrics.get("backtest") or {}).get("summary"),
            runtime_info=metrics.get("runtime"),
            status=status,
            errors=report_errors,
            historical_errors=metrics.get("historical_errors"),
        )
        write_report(run_dir / "report.md", report)

    (run_dir / "config.yaml").write_text(cfg_file.read_text())

    try:
        # Stage 00
        env = capture_env_snapshot(run_dir)
        metrics["env_snapshot_path"] = env.env_json_path
        state_payload["env_snapshot_path"] = env.env_json_path
        _save_state("stage_00_env_ready")
        _write_metrics()
        if stop_after_stage == "stage_00_env_ready":
            return run_id

        start_index = 0
        if reuse_run and last_state:
            last_success = last_success_stage(str(last_state.get("stage", "")))
            if last_success and last_success in STAGES_SUCCESS_ORDER:
                start_index = _safe_stage_index(last_success) + 1
                for s in STAGES_SUCCESS_ORDER[: _safe_stage_index(last_success) + 1]:
                    if not _stage_artifacts_exist(stage=s, paths=paths, run_dir=run_dir, state=state_payload, metrics=metrics):
                        start_index = _safe_stage_index(s)
                        break

        force_index = _safe_stage_index(force_stage) if force_stage in STAGES_SUCCESS_ORDER else None
        stop_index = _safe_stage_index(stop_after_stage) if stop_after_stage in STAGES_SUCCESS_ORDER else None

        def _should_run(stage: str) -> bool:
            idx = _safe_stage_index(stage)
            if force_index is not None and idx >= force_index:
                return True
            return idx >= start_index

        def _stop_if_needed(stage: str) -> bool:
            if stop_index is None:
                return False
            if _safe_stage_index(stage) >= stop_index:
                metrics["status"] = "partial"
                _write_metrics()
                return True
            return False

        # Stage 10
        if _should_run("stage_10_data_ready") or not (reuse_run and last_state and last_state.get("dataset_id")):
            fmp = FMPClient.from_env(paths.cache_http_dir, cfg_overrides=cfg.get("fmp", {}))
            dataset_id, universe, last_data_date = update_data(cfg, paths, fmp, force=False)
            metrics["dataset_id"] = dataset_id
            metrics["universe"] = {
                "provider_used": universe.provider_used,
                "fallback_used": universe.fallback_used,
                "fallback_reason": universe.fallback_reason,
                "end_date_truncated": universe.end_date_truncated,
                "start_date": universe.start_date,
                "end_date_effective": universe.end_date_effective,
                "symbols_count": len(universe.symbols_union),
            }
            metrics["last_data_date"] = last_data_date
            state_payload["dataset_id"] = dataset_id
            state_payload["last_data_date"] = last_data_date
        else:
            metrics["dataset_id"] = state_payload.get("dataset_id")
            metrics["last_data_date"] = state_payload.get("last_data_date")

        _save_state("stage_10_data_ready")
        _write_metrics()
        if _stop_if_needed("stage_10_data_ready"):
            return run_id

        dataset_id = str(metrics["dataset_id"])
        raw_dir = paths.raw_dir / dataset_id
        proc_dir = paths.processed_dir / dataset_id
        proc_dir.mkdir(parents=True, exist_ok=True)

        prices = pd.read_parquet(raw_dir / "prices.parquet")
        membership = pd.read_parquet(raw_dir / "universe_membership.parquet")
        earnings_path = raw_dir / "earnings.parquet"
        splits_path = raw_dir / "splits.parquet"
        dividends_path = raw_dir / "dividends.parquet"
        financials_path = raw_dir / "financials_quarterly.parquet"
        news_path = raw_dir / "news.parquet"
        macro_treasury_path = raw_dir / "macro_treasury.parquet"
        macro_calendar_path = raw_dir / "macro_calendar.parquet"
        analyst_path = raw_dir / "analyst_premium.parquet"
        earnings = pd.read_parquet(earnings_path) if earnings_path.exists() else pd.DataFrame()
        splits_df = pd.read_parquet(splits_path) if splits_path.exists() else pd.DataFrame()
        dividends_df = pd.read_parquet(dividends_path) if dividends_path.exists() else pd.DataFrame()
        financials = pd.read_parquet(financials_path) if financials_path.exists() else pd.DataFrame()
        include_news = bool((cfg.get("data") or {}).get("include_news", False))
        include_macro = bool((cfg.get("data") or {}).get("include_macro", True))
        include_analyst = bool((cfg.get("data") or {}).get("include_analyst", False))
        news: Optional[pd.DataFrame] = None
        if include_news:
            news = pd.read_parquet(news_path) if news_path.exists() else pd.DataFrame()
        macro_treasury: Optional[pd.DataFrame] = None
        macro_calendar: Optional[pd.DataFrame] = None
        if include_macro:
            macro_treasury = pd.read_parquet(macro_treasury_path) if macro_treasury_path.exists() else pd.DataFrame()
            macro_calendar = pd.read_parquet(macro_calendar_path) if macro_calendar_path.exists() else pd.DataFrame()
        analyst: Optional[pd.DataFrame] = None
        if include_analyst:
            analyst = pd.read_parquet(analyst_path) if analyst_path.exists() else pd.DataFrame()

        # Stage 20
        if _should_run("stage_20_validation_passed"):
            vcfg = (cfg.get("validation") or {}).get("thresholds", {}) or {}
            vres = validate_raw_prices(
                prices,
                adjusted_flag=bool((cfg.get("data") or {}).get("adjusted_flag", True)),
                thresholds=vcfg,
                splits=splits_df,
                dividends=dividends_df,
            )
            metrics["validation"] = vres.summary

            excluded = set(vres.excluded_symbols)
            if excluded:
                prices = prices[~prices["symbol"].astype(str).str.upper().isin(excluded)].reset_index(drop=True)
                membership = membership[~membership["symbol"].astype(str).str.upper().isin(excluded)].reset_index(drop=True)
                metrics["validation"]["excluded_symbols"] = sorted(excluded)

            prices.to_parquet(proc_dir / "prices_validated.parquet", index=False)
            membership.to_parquet(proc_dir / "universe_membership_validated.parquet", index=False)

            if not vres.passed:
                state_payload.update(append_state_error(state_payload, stage="stage_20_validation_failed", error=vres.summary.get("fail_reasons")))
                _save_state("stage_20_validation_failed")
                metrics["status"] = "failed_validation"
                _write_metrics()
                _write_report("FAILED", errors=state_payload["errors"])
                raise GateFailed("stage_20_validation_failed", "Validation gate failed")

        _save_state("stage_20_validation_passed")
        _write_metrics()
        if _stop_if_needed("stage_20_validation_passed"):
            return run_id

        # Stage 30
        if _should_run("stage_30_features_ready"):
            prices_in = pd.read_parquet(proc_dir / "prices_validated.parquet")
            mem_in = pd.read_parquet(proc_dir / "universe_membership_validated.parquet")

            fcfg = cfg.get("features") or {}
            feature_kwargs = dict(
                dataset_id=dataset_id,
                prices=prices_in,
                universe_membership=mem_in,
                earnings=earnings if not earnings.empty else None,
                adjusted_flag=bool((cfg.get("data") or {}).get("adjusted_flag", True)),
                lookbacks=list(fcfg.get("lookbacks", [1, 5, 20, 60])),
                event_safe_shift_days=int(fcfg.get("event_safe_shift_days", 1)),
                news_general_symbol_uses_tickers=bool(fcfg.get("news_general_symbol_uses_tickers", False)),
                out_dir=paths.feature_store_dir,
            )
            if "news" in inspect.signature(build_feature_store).parameters:
                feature_kwargs["news"] = news
            if "financials" in inspect.signature(build_feature_store).parameters:
                feature_kwargs["financials"] = financials if not financials.empty else None
            if "macro_treasury" in inspect.signature(build_feature_store).parameters:
                feature_kwargs["macro_treasury"] = macro_treasury
            if "macro_calendar" in inspect.signature(build_feature_store).parameters:
                feature_kwargs["macro_calendar"] = macro_calendar
            if "analyst" in inspect.signature(build_feature_store).parameters:
                feature_kwargs["analyst"] = analyst
            feature_res = build_feature_store(**feature_kwargs)
            metrics["feature_store_id"] = feature_res.feature_store_id
            state_payload["feature_store_id"] = feature_res.feature_store_id

            labels_res = build_future_return_labels(
                prices_in,
                adjusted_flag=bool((cfg.get("data") or {}).get("adjusted_flag", True)),
                horizon_trading_days=horizon,
            )
            labels = labels_res.labels

            mem_short = mem_in[mem_in["is_member"] == True][["date", "symbol"]].copy()  # noqa: E712
            mem_short["date"] = pd.to_datetime(mem_short["date"]).dt.date
            mem_short["symbol"] = mem_short["symbol"].astype(str).str.upper()
            labels = labels.merge(mem_short.rename(columns={"date": "decision_date"}), on=["decision_date", "symbol"], how="inner")
            labels.to_parquet(proc_dir / "labels.parquet", index=False)

            merged = feature_res.features.merge(labels, on=["decision_date", "symbol"], how="inner")
            merged.to_parquet(proc_dir / "merged.parquet", index=False)
            metrics["labels"] = labels_res.summary

        _save_state("stage_30_features_ready")
        _write_metrics()
        if _stop_if_needed("stage_30_features_ready"):
            return run_id

        feature_store_id = str(metrics.get("feature_store_id") or state_payload.get("feature_store_id") or "")
        merged = pd.read_parquet(proc_dir / "merged.parquet")

        # Stage 40
        if _should_run("stage_40_split_leakcheck_passed"):
            dts = sorted(pd.to_datetime(merged["decision_date"]).dt.date.unique().tolist())
            scfg = cfg.get("split") or {}
            wcfg = scfg.get("walkforward") or {}
            splits = make_walkforward_splits(
                dts,
                train=int(wcfg.get("train", 504)),
                val=int(wcfg.get("val", 126)),
                test=int(wcfg.get("test", 126)),
                step=int(wcfg.get("step", 63)),
            )
            if not splits:
                raise RuntimeError("Not enough dates for walkforward split")
            split = apply_purge_embargo(
                splits[-1],
                purge=int(scfg.get("purge", horizon)),
                embargo=int(scfg.get("embargo", 5)),
            )
            if not (split.train_dates and split.val_dates and split.test_dates):
                raise RuntimeError("Split empty after purge/embargo")

            features_df = pd.read_parquet(paths.feature_store_dir / feature_store_id / "features.parquet")
            labels_df = pd.read_parquet(proc_dir / "labels.parquet")
            leak = run_leakage_gate(
                features=features_df,
                labels=labels_df,
                split=split,
                target_col=target_col,
                corr_threshold=float((scfg.get("leakage") or {}).get("corr_threshold", 0.995)),
            )
            _write_json(run_dir / "leakage.json", leak)
            metrics["leakage"] = leak
            split_info = {
                "train_window": {"start": split.train_dates[0].isoformat(), "end": split.train_dates[-1].isoformat()},
                "val_window": {"start": split.val_dates[0].isoformat(), "end": split.val_dates[-1].isoformat()},
                "test_window": {"start": split.test_dates[0].isoformat(), "end": split.test_dates[-1].isoformat()},
            }
            metrics["split_info"] = split_info
            state_payload["train_window"] = split_info["train_window"]
            state_payload["val_window"] = split_info["val_window"]
            state_payload["test_window"] = split_info["test_window"]

            if not leak.get("passed", False):
                state_payload.update(append_state_error(state_payload, stage="stage_40_split_leakcheck_failed", error=leak.get("messages")))
                _save_state("stage_40_split_leakcheck_failed")
                metrics["status"] = "failed_leakage_gate"
                _write_metrics()
                _write_report("FAILED", errors=state_payload["errors"])
                raise GateFailed("stage_40_split_leakcheck_failed", "Leakage gate failed")

        _save_state("stage_40_split_leakcheck_passed")
        _write_metrics()
        if _stop_if_needed("stage_40_split_leakcheck_passed"):
            return run_id

        split_info = metrics.get("split_info") or {}
        merged["decision_date"] = pd.to_datetime(merged["decision_date"]).dt.date
        all_dates = sorted(merged["decision_date"].unique().tolist())

        def _dates_in_window(win: Dict[str, str]) -> list:
            s = pd.to_datetime(win["start"]).date()
            e = pd.to_datetime(win["end"]).date()
            return [d for d in all_dates if s <= d <= e]

        train_ds = _dates_in_window(split_info["train_window"])
        val_ds = _dates_in_window(split_info["val_window"])
        test_ds = _dates_in_window(split_info["test_window"])

        feature_cols = [
            c
            for c in merged.columns
            if c
            not in {
                "decision_date",
                "symbol",
                "label_date",
                "horizon",
                target_col,
                "feature_available_date",
                "feature_version",
            }
        ]
        fcfg = cfg.get("features") or {}
        exclude_prefixes = [str(x) for x in (fcfg.get("exclude_prefixes") or []) if str(x)]
        exclude_regex = [str(x) for x in (fcfg.get("exclude_regex") or []) if str(x)]
        feature_selection_meta: Dict[str, Any] = {
            "before_count": int(len(feature_cols)),
            "exclude_prefixes": exclude_prefixes,
            "exclude_regex": exclude_regex,
        }
        regexes = []
        for patt in exclude_regex:
            try:
                regexes.append(re.compile(patt))
            except re.error as exc:
                raise ValueError(f"Invalid features.exclude_regex pattern: {patt}") from exc

        if exclude_prefixes or regexes:
            before = list(feature_cols)

            def _keep_feature(name: str) -> bool:
                if exclude_prefixes and any(name.startswith(pfx) for pfx in exclude_prefixes):
                    return False
                if regexes and any(rx.search(name) for rx in regexes):
                    return False
                return True

            feature_cols = [c for c in before if _keep_feature(c)]
            feature_selection_meta["after_exclude_count"] = int(len(feature_cols))

        min_finite_ratio = max(0.0, float(fcfg.get("min_finite_ratio", 0.0) or 0.0))
        min_nonzero_ratio = max(0.0, float(fcfg.get("min_nonzero_ratio", 0.0) or 0.0))
        min_std = max(0.0, float(fcfg.get("min_std", 0.0) or 0.0))
        nonzero_eps = max(0.0, float(fcfg.get("nonzero_eps", 1e-12) or 1e-12))
        feature_cols, quality_summary = _apply_feature_quality_filter(
            merged=merged,
            feature_cols=feature_cols,
            train_dates=train_ds,
            min_finite_ratio=min_finite_ratio,
            min_nonzero_ratio=min_nonzero_ratio,
            min_std=min_std,
            nonzero_eps=nonzero_eps,
        )
        feature_selection_meta["quality_filter"] = quality_summary
        feature_selection_meta["after_count"] = int(len(feature_cols))
        metrics["feature_selection"] = feature_selection_meta
        if not feature_cols:
            raise RuntimeError("No features selected after applying features selection filters")

        # Stage 50
        if _should_run("stage_50_models_trained"):
            train_res = train_models(
                cfg=cfg,
                run_dir=run_dir,
                merged=merged,
                feature_cols=feature_cols,
                train_dates=train_ds,
                val_dates=val_ds,
            )
            state_payload["model_ckpt_paths"] = train_res.model_ckpt_paths
            metrics["training_info"] = train_res.training_info
            metrics["model_ckpt_paths"] = train_res.model_ckpt_paths

        _save_state("stage_50_models_trained")
        _write_metrics()
        if _stop_if_needed("stage_50_models_trained"):
            return run_id

        # Stage 60
        if _should_run("stage_60_predictions_ready"):
            from .models.gbdt import GBDTRegressor

            ckpts = state_payload.get("model_ckpt_paths") or metrics.get("model_ckpt_paths") or {}
            if "gbdt" not in ckpts:
                raise RuntimeError("GBDT checkpoint missing")
            model = GBDTRegressor.load(ckpts["gbdt"])

            rows = []
            for split_name, dset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
                part = merged[merged["decision_date"].isin(dset)].copy()
                X = part[feature_cols].to_numpy(dtype=float)
                y = part[target_col].to_numpy(dtype=float)
                keep = pd.notna(X).all(axis=1) & pd.notna(y)
                meta = part.loc[keep, ["decision_date", "symbol"] + [c for c in ["adv20_dollar", "vol_20d"] if c in part.columns]].copy()
                X = X[keep]
                y = y[keep]
                yhat = model.predict(X)
                out = meta.copy()
                out["model_name"] = "gbdt"
                out["split"] = split_name
                out["y_true"] = y
                out["y_pred"] = yhat
                rows.append(out)

            preds = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            preds.to_parquet(run_dir / "predictions.parquet", index=False)
            metrics["model_metrics"] = evaluate_predictions(preds)
            metrics["backtest"] = {"chosen_model": "gbdt"}

        _save_state("stage_60_predictions_ready")
        _write_metrics()
        _run_auto_review("stage_60_predictions_ready")
        _write_metrics()
        if _stop_if_needed("stage_60_predictions_ready"):
            return run_id

        # Stage 70
        if _should_run("stage_70_backtest_ready"):
            preds = pd.read_parquet(run_dir / "predictions.parquet")
            test_preds = preds[(preds["model_name"] == "gbdt") & (preds["split"] == "test")].copy()
            btcfg = cfg.get("backtest") or {}

            cm = cfg.get("cost_model")
            if isinstance(cm, str) and cm:
                cost_cfg = load_config(project_dir / cm if not Path(cm).is_absolute() else Path(cm))
            elif isinstance(cm, dict):
                cost_cfg = cm
            else:
                cost_cfg = {}

            result = run_backtest(
                test_preds,
                topn=int(btcfg.get("topn", 50)),
                long_short=bool(btcfg.get("long_short", False)),
                max_names=int(btcfg.get("max_names", 50)),
                single_name_cap=float(btcfg.get("single_name_cap", 0.05)),
                rebalance=str(btcfg.get("rebalance", "daily")),
                commission_bps=float(cost_cfg.get("commission_bps", 0.0)),
                slippage_base_bps=float((cost_cfg.get("slippage") or {}).get("base_bps", 0.0)),
                slippage_k_adv=float((cost_cfg.get("slippage") or {}).get("k_adv", 0.0)),
                slippage_k_vol=float((cost_cfg.get("slippage") or {}).get("k_vol", 0.0)),
                notional=float(btcfg.get("notional", 1_000_000.0)),
            )
            result.daily.to_parquet(run_dir / "backtest.parquet", index=False)
            result.positions.to_parquet(run_dir / "backtest_positions.parquet", index=False)
            metrics["backtest"] = {"chosen_model": "gbdt", "summary": result.summary}

        _save_state("stage_70_backtest_ready")
        _write_metrics()
        _run_auto_review("stage_70_backtest_ready")
        _write_metrics()
        if _stop_if_needed("stage_70_backtest_ready"):
            return run_id

        # Stage 80
        if isinstance(metrics.get("errors"), list) and metrics["errors"]:
            hist = metrics.get("historical_errors") if isinstance(metrics.get("historical_errors"), list) else []
            metrics["historical_errors"] = (hist + metrics["errors"])[-200:]
            metrics["errors"] = []
        _finalize_gpu_sampling()
        metrics["status"] = "success"
        _write_metrics()
        _write_report("SUCCESS")
        _run_auto_review("stage_80_report_ready")
        _write_metrics()
        _save_state("stage_80_report_ready")
        _write_metrics()
        return run_id

    except GateFailed:
        raise
    except Exception as exc:
        logger.exception("pipeline_failed")
        _finalize_gpu_sampling()
        err = _summarize_error(exc)
        err["stage"] = state_payload.get("stage")
        metrics["status"] = "error"
        metrics.setdefault("errors", []).append(err)
        state_payload.update(append_state_error(state_payload, stage=str(state_payload.get("stage")), error=err))
        _write_metrics()
        _write_report("ERROR", errors=state_payload.get("errors"))
        _save_state(str(state_payload.get("stage", "stage_00_env_ready")))

        err_dir = paths.artifacts_dir / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        _write_json(err_dir / f"{_utc_ts()}_{run_id}.json", {"run_id": run_id, "error": err})
        raise
    finally:
        try:
            _finalize_gpu_sampling()
            _write_metrics()
        except Exception:
            pass
        try:
            release_lock(paths.state_dir)
        except Exception:
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    p.add_argument("--conf", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--allow-conf-mismatch-resume", action="store_true")
    p.add_argument("--force-stage", default=None)
    p.add_argument("--force-unlock", action="store_true")
    p.add_argument("--stop-after-stage", default=None)
    args = p.parse_args(argv)

    run_pipeline(
        project_dir=Path(args.project_dir),
        conf_path=Path(args.conf),
        resume=bool(args.resume),
        resume_run_id=args.run_id,
        allow_resume_conf_mismatch=bool(args.allow_conf_mismatch_resume),
        force_stage=args.force_stage,
        force_unlock=bool(args.force_unlock),
        stop_after_stage=args.stop_after_stage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
