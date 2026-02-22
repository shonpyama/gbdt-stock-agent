#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gbdt_agent.operations import collect_ops_status, evaluate_ops_gate, load_ops_policy
from gbdt_agent.orchestrator import run_pipeline
from gbdt_agent.paths import ProjectPaths


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _metric_mean(metrics: Dict[str, Any], *, split: str, key: str) -> float:
    block = (((metrics.get("model_metrics") or {}).get("gbdt") or {}).get(split) or {})
    return _safe_float(((block.get(key) or {}).get("mean")))


def _metric_n(metrics: Dict[str, Any], *, split: str, key: str) -> int:
    block = (((metrics.get("model_metrics") or {}).get("gbdt") or {}).get(split) or {})
    try:
        return int(((block.get(key) or {}).get("n")) or 0)
    except Exception:
        return 0


def _overfit_diagnostics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    train_rank_ic = _metric_mean(metrics, split="train", key="rank_ic")
    val_rank_ic = _metric_mean(metrics, split="val", key="rank_ic")
    test_rank_ic = _metric_mean(metrics, split="test", key="rank_ic")
    train_ic = _metric_mean(metrics, split="train", key="ic")
    val_ic = _metric_mean(metrics, split="val", key="ic")
    test_ic = _metric_mean(metrics, split="test", key="ic")
    val_days = _metric_n(metrics, split="val", key="rank_ic")
    test_days = _metric_n(metrics, split="test", key="rank_ic")

    gap_rank = 0.0
    if train_rank_ic == train_rank_ic and val_rank_ic == val_rank_ic:
        gap_rank = max(0.0, train_rank_ic - val_rank_ic)
    gap_ic = 0.0
    if train_ic == train_ic and val_ic == val_ic:
        gap_ic = max(0.0, train_ic - val_ic)

    # Select on validation signal and penalize excessive train/val divergence.
    val_rank_component = val_rank_ic if val_rank_ic == val_rank_ic else -1.0
    val_ic_component = val_ic if val_ic == val_ic else -1.0
    low_val_days_penalty = max(0, 10 - int(val_days)) * 0.75
    selection_score = (val_rank_component * 120.0) + (val_ic_component * 40.0) - (gap_rank * 110.0) - (gap_ic * 30.0) - low_val_days_penalty
    overfit_risk_high = bool((gap_rank > 0.25) or (gap_ic > 0.20))
    if overfit_risk_high:
        selection_score -= 8.0
    return {
        "train_rank_ic_mean": train_rank_ic,
        "val_rank_ic_mean": val_rank_ic,
        "test_rank_ic_mean": test_rank_ic,
        "train_ic_mean": train_ic,
        "val_ic_mean": val_ic,
        "test_ic_mean": test_ic,
        "val_n_days": int(val_days),
        "test_n_days": int(test_days),
        "overfit_gap_rank_ic": gap_rank,
        "overfit_gap_ic": gap_ic,
        "overfit_risk_high": overfit_risk_high,
        "selection_score": float(selection_score),
    }


def _test_monitor_score(metrics: Dict[str, Any]) -> float:
    model = ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test") or {}
    rank_ic = _safe_float(((model.get("rank_ic") or {}).get("mean")))
    back = ((metrics.get("backtest") or {}).get("summary") or {})
    sharpe = _safe_float(back.get("sharpe"))
    total_ret = _safe_float(back.get("total_return"))
    max_dd = _safe_float(back.get("max_drawdown"))
    if rank_ic != rank_ic:
        rank_ic = -1.0
    if sharpe != sharpe:
        sharpe = -10.0
    if total_ret != total_ret:
        total_ret = -10.0
    if max_dd != max_dd:
        max_dd = -1.0
    return (rank_ic * 120.0) + (sharpe * 1.2) + (total_ret * 0.05) + (max_dd * 4.0)


def _candidate_map() -> List[Dict[str, Any]]:
    return [
        {"name": "baseline_auto", "params": {}, "overrides": {}},
        {
            "name": "compact_31",
            "params": {
                "n_estimators": 1500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
            },
            "overrides": {},
        },
        {
            "name": "regularized_63",
            "params": {
                "n_estimators": 5000,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "subsample": 0.85,
                "colsample_bytree": 0.7,
                "min_child_samples": 40,
                "reg_lambda": 2.0,
                "reg_alpha": 0.1,
            },
            "overrides": {},
        },
        {
            "name": "ultra_regularized_31",
            "params": {
                "n_estimators": 2200,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.65,
                "min_child_samples": 60,
                "reg_lambda": 4.0,
                "reg_alpha": 0.2,
            },
            "overrides": {},
        },
        {
            "name": "max_regularized_15",
            "params": {
                "n_estimators": 2600,
                "learning_rate": 0.025,
                "num_leaves": 15,
                "subsample": 0.7,
                "colsample_bytree": 0.55,
                "min_child_samples": 100,
                "reg_lambda": 8.0,
                "reg_alpha": 0.4,
            },
            "overrides": {},
        },
        {
            "name": "baseline_auto_long_weekly_3",
            "params": {},
            "overrides": {
                "backtest": {
                    "long_short": False,
                    "topn": 3,
                    "max_names": 3,
                    "single_name_cap": 0.3,
                    "rebalance": "weekly",
                }
            },
        },
        {
            "name": "compact_31_ls_daily_3x8",
            "params": {
                "n_estimators": 1500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
            },
            "overrides": {
                "backtest": {
                    "long_short": True,
                    "topn": 3,
                    "max_names": 8,
                    "single_name_cap": 0.3,
                    "rebalance": "daily",
                }
            },
        },
    ]


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _resolve_conf_path(base: str) -> Path:
    path = Path(base)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run production model search and rank candidate params.")
    parser.add_argument("--base-conf", default="conf/default.yaml", help="Base config path used as a template.")
    parser.add_argument(
        "--only-candidates",
        default="",
        help="Comma-separated candidate names to run (default: all).",
    )
    parser.add_argument("--out-json", default="", help="Optional output JSON path.")
    parser.add_argument("--out-md", default="", help="Optional output Markdown path.")
    parser.add_argument(
        "--promote-default",
        action="store_true",
        help="Apply selected model params to base config after ranking.",
    )
    parser.add_argument(
        "--skip-materialize-selected",
        action="store_true",
        help="Do not re-run the selected candidate at the end (default: re-run so last_run_state matches selection).",
    )
    args = parser.parse_args()

    project_dir = PROJECT_DIR
    paths = ProjectPaths.from_project_dir(project_dir)
    base_conf_path = _resolve_conf_path(args.base_conf)
    base_cfg = yaml.safe_load(base_conf_path.read_text())
    policy_path = project_dir / "conf" / "ops_policy.yaml"
    policy = load_ops_policy(policy_path)
    max_age_hours = float(policy.get("max_age_hours", 72.0))
    require_gpu = bool(policy.get("require_gpu", False))

    candidates = _candidate_map()
    only_raw = str(args.only_candidates or "").strip()
    if only_raw:
        allow = {x.strip() for x in only_raw.split(",") if x.strip()}
        candidates = [c for c in candidates if c["name"] in allow]
        if not candidates:
            raise ValueError(f"No candidates matched --only-candidates={only_raw!r}")

    out_dir = project_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = project_dir / "conf" / "experiments"
    conf_dir.mkdir(parents=True, exist_ok=True)

    explicit_end = str(os.environ.get("MODEL_SEARCH_END_DATE", "")).strip()
    default_end = (date.today() - timedelta(days=1)).isoformat()
    selected_end_date = explicit_end or default_end

    warm_seed: Dict[str, Any] = {"run_id": None, "error": ""}
    if candidates:
        try:
            warm_cfg = deepcopy(base_cfg)
            warm_cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = dict(candidates[0].get("params") or {})
            warm_cfg.setdefault("data", {})["end_date"] = selected_end_date
            warm_cfg.setdefault("run", {})["log_level"] = "WARNING"
            warm_conf_path = conf_dir / f"prod_model_warm_stage40_{selected_end_date.replace('-', '')}.yaml"
            warm_conf_path.write_text(yaml.safe_dump(warm_cfg, sort_keys=False))
            warm_seed["run_id"] = run_pipeline(
                project_dir=project_dir,
                conf_path=warm_conf_path,
                resume=False,
                stop_after_stage="stage_40_split_leakcheck_passed",
                force_unlock=True,
            )
        except Exception as exc:
            warm_seed["error"] = f"{type(exc).__name__}: {exc}"

    results: List[Dict[str, Any]] = []
    for idx, c in enumerate(candidates, start=1):
        cfg = deepcopy(base_cfg)
        cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = dict(c.get("params") or {})
        overrides = deepcopy(c.get("overrides") or {})
        if overrides:
            _deep_update(cfg, overrides)
        cfg.setdefault("data", {})["end_date"] = selected_end_date
        # Keep logs concise during sweep runs.
        cfg.setdefault("run", {})["log_level"] = "WARNING"
        conf_path = conf_dir / f"prod_model_{idx:02d}_{c['name']}.yaml"
        conf_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        item: Dict[str, Any] = {
            "name": c["name"],
            "conf_path": str(conf_path),
            "params": dict(c.get("params") or {}),
            "overrides": overrides,
            "ok": False,
            "warm_start_used": bool(warm_seed.get("run_id")),
            "warm_source_run_id": warm_seed.get("run_id"),
        }
        if warm_seed.get("error"):
            item["warm_start_error"] = warm_seed["error"]
        try:
            if warm_seed.get("run_id"):
                run_id = run_pipeline(
                    project_dir=project_dir,
                    conf_path=conf_path,
                    resume=True,
                    resume_run_id=str(warm_seed["run_id"]),
                    allow_resume_conf_mismatch=True,
                    force_stage="stage_50_models_trained",
                    force_unlock=True,
                )
            else:
                run_id = run_pipeline(
                    project_dir=project_dir,
                    conf_path=conf_path,
                    resume=False,
                    force_unlock=True,
                )
            metrics_path = paths.run_dir(run_id) / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            status_payload = collect_ops_status(
                project_dir=project_dir,
                run_id=run_id,
                max_age_hours=max_age_hours,
                require_gpu=require_gpu,
            )
            gate_payload = evaluate_ops_gate(status_payload, policy)

            item["run_id"] = run_id
            item["ok"] = bool(metrics.get("status") == "success")
            item["status"] = metrics.get("status")
            item["ops_gate_ok"] = bool(gate_payload.get("ok"))
            item["ops_gate_violations"] = gate_payload.get("violations", [])

            mm = ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test") or {}
            bt = ((metrics.get("backtest") or {}).get("summary") or {})
            diag = _overfit_diagnostics(metrics)
            item.update(diag)
            item["rank_ic_test_mean"] = ((mm.get("rank_ic") or {}).get("mean"))
            item["ic_test_mean"] = ((mm.get("ic") or {}).get("mean"))
            item["sharpe"] = bt.get("sharpe")
            item["total_return"] = bt.get("total_return")
            item["max_drawdown"] = bt.get("max_drawdown")
            item["avg_turnover"] = bt.get("avg_turnover")
            item["avg_cost"] = bt.get("avg_cost")
            item["test_monitor_score"] = _test_monitor_score(metrics)
            # Backward-compatible key used by ranking/readiness scripts.
            item["score"] = float(item.get("selection_score", float("-inf")))
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    ranked = sorted(results, key=lambda x: float(x.get("score", -1e9)), reverse=True)
    gated_ranked = [r for r in ranked if bool(r.get("ops_gate_ok"))]
    gated_low_overfit = [r for r in gated_ranked if not bool(r.get("overfit_risk_high"))]
    selected = gated_low_overfit[0] if gated_low_overfit else (gated_ranked[0] if gated_ranked else (ranked[0] if ranked else None))
    materialized_selected_run_id = None
    materialized_selected_error = ""
    if selected and not bool(args.skip_materialize_selected):
        try:
            selected_conf = Path(str(selected.get("conf_path")))
            if not selected_conf.is_absolute():
                selected_conf = project_dir / selected_conf

            # Keep runtime state aligned with the selected candidate.
            if warm_seed.get("run_id"):
                materialized_selected_run_id = run_pipeline(
                    project_dir=project_dir,
                    conf_path=selected_conf,
                    resume=True,
                    resume_run_id=str(warm_seed["run_id"]),
                    allow_resume_conf_mismatch=True,
                    force_stage="stage_50_models_trained",
                    force_unlock=True,
                )
            else:
                materialized_selected_run_id = run_pipeline(
                    project_dir=project_dir,
                    conf_path=selected_conf,
                    resume=False,
                    force_unlock=True,
                )

            selected = dict(selected)
            selected["materialized_run_id"] = materialized_selected_run_id
            selected["run_id"] = materialized_selected_run_id
        except Exception as exc:
            materialized_selected_error = f"{type(exc).__name__}: {exc}"

    payload = {
        "base_conf": str(base_conf_path),
        "policy_path": str(policy_path),
        "selected_end_date": selected_end_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warm_seed": warm_seed,
        "materialized_selected_run_id": materialized_selected_run_id,
        "materialized_selected_error": materialized_selected_error,
        "ranked": ranked,
        "selected": selected,
    }
    out_json = Path(args.out_json) if args.out_json else (out_dir / "model_search_prod_results.json")
    out_md = Path(args.out_md) if args.out_md else (out_dir / f"model_search_prod_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md")
    if not out_json.is_absolute():
        out_json = project_dir / out_json
    if not out_md.is_absolute():
        out_md = project_dir / out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    lines = [
        f"# Model Search (Production) - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        f"- base_conf: `{base_conf_path}`",
        f"- policy: `{policy_path}`",
        f"- selected_end_date: `{selected_end_date}`",
        f"- trials: `{len(results)}`",
        "",
        "| rank | name | selection_score | val_rank_ic | train_val_gap_rank_ic | test_rank_ic | sharpe | max_drawdown | overfit_risk_high | ops_gate | run_id |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        gate_txt = "pass" if bool(r.get("ops_gate_ok")) else "fail"
        overfit_txt = "high" if bool(r.get("overfit_risk_high")) else "low"
        lines.append(
            f"| {i} | {r.get('name')} | {r.get('selection_score')} | {r.get('val_rank_ic_mean')} | {r.get('overfit_gap_rank_ic')} | {r.get('rank_ic_test_mean')} | {r.get('sharpe')} | {r.get('max_drawdown')} | {overfit_txt} | {gate_txt} | {r.get('run_id','-')} |"
        )
    if selected:
        lines += [
            "",
            "## Selected",
            "",
            f"- name: `{selected.get('name')}`",
            f"- run_id: `{selected.get('run_id')}`",
            f"- selection_score: `{selected.get('selection_score')}`",
            f"- val_rank_ic_mean: `{selected.get('val_rank_ic_mean')}`",
            f"- overfit_gap_rank_ic: `{selected.get('overfit_gap_rank_ic')}`",
            f"- overfit_gap_ic: `{selected.get('overfit_gap_ic')}`",
            f"- overfit_risk_high: `{selected.get('overfit_risk_high')}`",
            f"- test_monitor_score: `{selected.get('test_monitor_score')}`",
            f"- params: `{json.dumps(selected.get('params', {}), ensure_ascii=True)}`",
            f"- overrides: `{json.dumps(selected.get('overrides', {}), ensure_ascii=True)}`",
        ]
    out_md.write_text("\n".join(lines) + "\n")

    promoted = False
    if args.promote_default and selected:
        promoted_cfg = yaml.safe_load(base_conf_path.read_text())
        promoted_cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = dict(selected.get("params") or {})
        selected_overrides = selected.get("overrides") if isinstance(selected.get("overrides"), dict) else {}
        if selected_overrides:
            _deep_update(promoted_cfg, deepcopy(selected_overrides))
        base_conf_path.write_text(yaml.safe_dump(promoted_cfg, sort_keys=False))
        promoted = True

    print(
        json.dumps(
            {
                "results_json": str(out_json),
                "results_md": str(out_md),
                "selected": selected,
                "top": ranked[:3],
                "promoted_default": promoted,
                "base_conf": str(base_conf_path),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
