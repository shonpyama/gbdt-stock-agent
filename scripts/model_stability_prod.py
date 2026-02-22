#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
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


def _parse_end_dates() -> List[str]:
    raw = str(os.environ.get("MODEL_STABILITY_END_DATES", "")).strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return ["2025-12-31", "2026-01-31", "2026-02-21"]


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
    ]


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _mean(xs: List[float]) -> float:
    vals = [x for x in xs if x == x]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _resolve_conf_path(base: str) -> Path:
    path = Path(base)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-period stability validation for production model selection.")
    parser.add_argument(
        "--base-conf",
        default="conf/default.yaml",
        help="Base config path used as a template.",
    )
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
    args = parser.parse_args()

    project_dir = PROJECT_DIR
    paths = ProjectPaths.from_project_dir(project_dir)
    base_conf_path = _resolve_conf_path(args.base_conf)
    base_cfg = yaml.safe_load(base_conf_path.read_text())

    policy_path = project_dir / "conf" / "ops_policy.yaml"
    policy = load_ops_policy(policy_path)
    max_age_hours = float(policy.get("max_age_hours", 72.0))
    require_gpu = bool(policy.get("require_gpu", False))

    end_dates = _parse_end_dates()
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

    results: List[Dict[str, Any]] = []
    warm_seed_by_end_date: Dict[str, Any] = {}
    for end_date in end_dates:
        end_tag = end_date.replace("-", "")
        warm_run_id = None
        warm_error = ""
        if candidates:
            try:
                warm_cfg = deepcopy(base_cfg)
                warm_cfg.setdefault("data", {})["end_date"] = end_date
                warm_cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = dict(candidates[0].get("params") or {})
                warm_cfg.setdefault("run", {})["log_level"] = "WARNING"
                warm_conf_path = conf_dir / f"stability_model_{end_tag}_warm_stage40.yaml"
                warm_conf_path.write_text(yaml.safe_dump(warm_cfg, sort_keys=False))
                warm_run_id = run_pipeline(
                    project_dir=project_dir,
                    conf_path=warm_conf_path,
                    resume=False,
                    stop_after_stage="stage_40_split_leakcheck_passed",
                    force_unlock=True,
                )
            except Exception as exc:
                warm_error = f"{type(exc).__name__}: {exc}"
        warm_seed_by_end_date[end_date] = {"run_id": warm_run_id, "error": warm_error}

        for c in candidates:
            cfg = deepcopy(base_cfg)
            cfg.setdefault("data", {})["end_date"] = end_date
            cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = dict(c.get("params") or {})
            overrides = deepcopy(c.get("overrides") or {})
            if overrides:
                _deep_update(cfg, overrides)
            cfg.setdefault("run", {})["log_level"] = "WARNING"

            conf_path = conf_dir / f"stability_model_{end_tag}_{c['name']}.yaml"
            conf_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

            item: Dict[str, Any] = {
                "end_date": end_date,
                "name": c["name"],
                "params": dict(c.get("params") or {}),
                "overrides": overrides,
                "conf_path": str(conf_path),
                "ok": False,
                "warm_start_used": bool(warm_run_id),
                "warm_source_run_id": warm_run_id,
            }
            if warm_error:
                item["warm_start_error"] = warm_error
            try:
                if warm_run_id:
                    run_id = run_pipeline(
                        project_dir=project_dir,
                        conf_path=conf_path,
                        resume=True,
                        resume_run_id=str(warm_run_id),
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

                mm = ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test") or {}
                bt = ((metrics.get("backtest") or {}).get("summary") or {})

                item["run_id"] = run_id
                item["ok"] = bool(metrics.get("status") == "success")
                item["status"] = metrics.get("status")
                item["ops_gate_ok"] = bool(gate_payload.get("ok"))
                item["ops_gate_violations"] = gate_payload.get("violations", [])
                item.update(_overfit_diagnostics(metrics))
                item["rank_ic_test_mean"] = ((mm.get("rank_ic") or {}).get("mean"))
                item["ic_test_mean"] = ((mm.get("ic") or {}).get("mean"))
                item["sharpe"] = bt.get("sharpe")
                item["total_return"] = bt.get("total_return")
                item["max_drawdown"] = bt.get("max_drawdown")
                item["avg_turnover"] = bt.get("avg_turnover")
                item["avg_cost"] = bt.get("avg_cost")
                # Backward-compatible key used by ranking/readiness scripts.
                item["score"] = float(item.get("selection_score", float("-inf")))
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
            results.append(item)

    summary: List[Dict[str, Any]] = []
    for c in candidates:
        rows = [r for r in results if r.get("name") == c["name"]]
        summary.append(
            {
                "name": c["name"],
                "params": dict(c.get("params") or {}),
                "overrides": deepcopy(c.get("overrides") or {}),
                "periods": len(rows),
                "gate_pass_periods": sum(1 for r in rows if bool(r.get("ops_gate_ok"))),
                "all_periods_gate_pass": all(bool(r.get("ops_gate_ok")) for r in rows) and len(rows) == len(end_dates),
                "score_mean": _mean([_safe_float(r.get("score")) for r in rows]),
                "score_min": min((_safe_float(r.get("score")) for r in rows), default=float("nan")),
                "val_rank_ic_mean": _mean([_safe_float(r.get("val_rank_ic_mean")) for r in rows]),
                "rank_ic_mean": _mean([_safe_float(r.get("rank_ic_test_mean")) for r in rows]),
                "sharpe_mean": _mean([_safe_float(r.get("sharpe")) for r in rows]),
                "total_return_mean": _mean([_safe_float(r.get("total_return")) for r in rows]),
                "max_drawdown_mean": _mean([_safe_float(r.get("max_drawdown")) for r in rows]),
                "overfit_risk_periods": int(sum(1 for r in rows if bool(r.get("overfit_risk_high")))),
            }
        )

    ranked_summary = sorted(
        summary,
        key=lambda x: (
            1 if bool(x.get("all_periods_gate_pass")) else 0,
            _safe_float(x.get("score_mean")),
            _safe_float(x.get("score_min")),
        ),
        reverse=True,
    )
    selected = ranked_summary[0] if ranked_summary else None

    payload = {
        "base_conf": str(base_conf_path),
        "policy_path": str(policy_path),
        "end_dates": end_dates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warm_seed_by_end_date": warm_seed_by_end_date,
        "results": results,
        "summary": ranked_summary,
        "selected": selected,
    }
    out_json = Path(args.out_json) if args.out_json else (out_dir / "model_stability_prod_results.json")
    out_md = Path(args.out_md) if args.out_md else (out_dir / f"model_stability_prod_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md")
    if not out_json.is_absolute():
        out_json = project_dir / out_json
    if not out_md.is_absolute():
        out_md = project_dir / out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    lines = [
        f"# Model Stability (Production) - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        f"- base_conf: `{base_conf_path}`",
        f"- policy: `{policy_path}`",
        f"- end_dates: `{json.dumps(end_dates, ensure_ascii=True)}`",
        "",
        "## Per-Period",
        "",
        "| end_date | model | selection_score | val_rank_ic | train_val_gap_rank_ic | test_rank_ic | sharpe | max_drawdown | overfit_risk_high | ops_gate | run_id |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: (str(x.get("end_date")), str(x.get("name")))):
        gate_txt = "pass" if bool(r.get("ops_gate_ok")) else "fail"
        overfit_txt = "high" if bool(r.get("overfit_risk_high")) else "low"
        lines.append(
            f"| {r.get('end_date')} | {r.get('name')} | {r.get('selection_score')} | {r.get('val_rank_ic_mean')} | {r.get('overfit_gap_rank_ic')} | {r.get('rank_ic_test_mean')} | {r.get('sharpe')} | {r.get('max_drawdown')} | {overfit_txt} | {gate_txt} | {r.get('run_id','-')} |"
        )

    lines += [
        "",
        "## Aggregate",
        "",
        "| rank | model | all_periods_gate_pass | score_mean | score_min | val_rank_ic_mean | test_rank_ic_mean | sharpe_mean | total_return_mean | max_drawdown_mean | overfit_risk_periods |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, s in enumerate(ranked_summary, start=1):
        lines.append(
            f"| {idx} | {s.get('name')} | {s.get('all_periods_gate_pass')} | {s.get('score_mean')} | {s.get('score_min')} | {s.get('val_rank_ic_mean')} | {s.get('rank_ic_mean')} | {s.get('sharpe_mean')} | {s.get('total_return_mean')} | {s.get('max_drawdown_mean')} | {s.get('overfit_risk_periods')} |"
        )

    if selected:
        lines += [
            "",
            "## Selected",
            "",
            f"- name: `{selected.get('name')}`",
            f"- params: `{json.dumps(selected.get('params', {}), ensure_ascii=True)}`",
            f"- overrides: `{json.dumps(selected.get('overrides', {}), ensure_ascii=True)}`",
            f"- all_periods_gate_pass: `{selected.get('all_periods_gate_pass')}`",
            f"- score_mean: `{selected.get('score_mean')}`",
            f"- score_min: `{selected.get('score_min')}`",
            f"- val_rank_ic_mean: `{selected.get('val_rank_ic_mean')}`",
            f"- test_rank_ic_mean: `{selected.get('rank_ic_mean')}`",
            f"- overfit_risk_periods: `{selected.get('overfit_risk_periods')}`",
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
