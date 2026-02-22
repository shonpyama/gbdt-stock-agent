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


def _score(metrics: Dict[str, Any]) -> float:
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


def _parse_end_dates() -> List[str]:
    raw = str(os.environ.get("MODEL_STABILITY_END_DATES", "")).strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return ["2025-12-31", "2026-01-31", "2026-02-21"]


def _candidate_map() -> List[Dict[str, Any]]:
    return [
        {"name": "baseline_auto", "params": {}},
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
        },
    ]


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
    for end_date in end_dates:
        for c in candidates:
            cfg = deepcopy(base_cfg)
            cfg.setdefault("data", {})["end_date"] = end_date
            cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = dict(c["params"])
            cfg.setdefault("run", {})["log_level"] = "WARNING"

            end_tag = end_date.replace("-", "")
            conf_path = conf_dir / f"stability_model_{end_tag}_{c['name']}.yaml"
            conf_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

            item: Dict[str, Any] = {
                "end_date": end_date,
                "name": c["name"],
                "params": dict(c["params"]),
                "conf_path": str(conf_path),
                "ok": False,
            }
            try:
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
                item["rank_ic_test_mean"] = ((mm.get("rank_ic") or {}).get("mean"))
                item["ic_test_mean"] = ((mm.get("ic") or {}).get("mean"))
                item["sharpe"] = bt.get("sharpe")
                item["total_return"] = bt.get("total_return")
                item["max_drawdown"] = bt.get("max_drawdown")
                item["avg_turnover"] = bt.get("avg_turnover")
                item["avg_cost"] = bt.get("avg_cost")
                item["score"] = _score(metrics)
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
            results.append(item)

    summary: List[Dict[str, Any]] = []
    for c in candidates:
        rows = [r for r in results if r.get("name") == c["name"]]
        summary.append(
            {
                "name": c["name"],
                "params": dict(c["params"]),
                "periods": len(rows),
                "gate_pass_periods": sum(1 for r in rows if bool(r.get("ops_gate_ok"))),
                "all_periods_gate_pass": all(bool(r.get("ops_gate_ok")) for r in rows) and len(rows) == len(end_dates),
                "score_mean": _mean([_safe_float(r.get("score")) for r in rows]),
                "score_min": min((_safe_float(r.get("score")) for r in rows), default=float("nan")),
                "rank_ic_mean": _mean([_safe_float(r.get("rank_ic_test_mean")) for r in rows]),
                "sharpe_mean": _mean([_safe_float(r.get("sharpe")) for r in rows]),
                "total_return_mean": _mean([_safe_float(r.get("total_return")) for r in rows]),
                "max_drawdown_mean": _mean([_safe_float(r.get("max_drawdown")) for r in rows]),
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
        "| end_date | model | score | rank_ic_test_mean | sharpe | total_return | max_drawdown | ops_gate | run_id |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(results, key=lambda x: (str(x.get("end_date")), str(x.get("name")))):
        gate_txt = "pass" if bool(r.get("ops_gate_ok")) else "fail"
        lines.append(
            f"| {r.get('end_date')} | {r.get('name')} | {r.get('score')} | {r.get('rank_ic_test_mean')} | {r.get('sharpe')} | {r.get('total_return')} | {r.get('max_drawdown')} | {gate_txt} | {r.get('run_id','-')} |"
        )

    lines += [
        "",
        "## Aggregate",
        "",
        "| rank | model | all_periods_gate_pass | score_mean | score_min | rank_ic_mean | sharpe_mean | total_return_mean | max_drawdown_mean |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, s in enumerate(ranked_summary, start=1):
        lines.append(
            f"| {idx} | {s.get('name')} | {s.get('all_periods_gate_pass')} | {s.get('score_mean')} | {s.get('score_min')} | {s.get('rank_ic_mean')} | {s.get('sharpe_mean')} | {s.get('total_return_mean')} | {s.get('max_drawdown_mean')} |"
        )

    if selected:
        lines += [
            "",
            "## Selected",
            "",
            f"- name: `{selected.get('name')}`",
            f"- params: `{json.dumps(selected.get('params', {}), ensure_ascii=True)}`",
            f"- all_periods_gate_pass: `{selected.get('all_periods_gate_pass')}`",
            f"- score_mean: `{selected.get('score_mean')}`",
            f"- score_min: `{selected.get('score_min')}`",
        ]
    out_md.write_text("\n".join(lines) + "\n")

    promoted = False
    if args.promote_default and selected:
        promoted_cfg = yaml.safe_load(base_conf_path.read_text())
        promoted_cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = dict(selected.get("params") or {})
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
