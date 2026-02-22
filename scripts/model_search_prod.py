#!/usr/bin/env python3
from __future__ import annotations

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


def _score(metrics: Dict[str, Any]) -> float:
    model = ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test") or {}
    rank_ic = _safe_float(((model.get("rank_ic") or {}).get("mean"))
)
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
    # Prioritize ranking quality and risk-adjusted returns.
    return (rank_ic * 120.0) + (sharpe * 1.2) + (total_ret * 0.05) + (max_dd * 4.0)


def main() -> int:
    project_dir = PROJECT_DIR
    paths = ProjectPaths.from_project_dir(project_dir)
    base_conf_path = project_dir / "conf" / "default.yaml"
    base_cfg = yaml.safe_load(base_conf_path.read_text())
    policy_path = project_dir / "conf" / "ops_policy.yaml"
    policy = load_ops_policy(policy_path)
    max_age_hours = float(policy.get("max_age_hours", 72.0))
    require_gpu = bool(policy.get("require_gpu", False))

    candidates: List[Dict[str, Any]] = [
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
        },
    ]

    out_dir = project_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = project_dir / "conf" / "experiments"
    conf_dir.mkdir(parents=True, exist_ok=True)

    explicit_end = str(os.environ.get("MODEL_SEARCH_END_DATE", "")).strip()
    default_end = (date.today() - timedelta(days=1)).isoformat()
    selected_end_date = explicit_end or default_end

    results: List[Dict[str, Any]] = []
    for idx, c in enumerate(candidates, start=1):
        cfg = deepcopy(base_cfg)
        cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = dict(c["params"])
        cfg.setdefault("data", {})["end_date"] = selected_end_date
        # Keep logs concise during sweep runs.
        cfg.setdefault("run", {})["log_level"] = "WARNING"
        conf_path = conf_dir / f"prod_model_{idx:02d}_{c['name']}.yaml"
        conf_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        item: Dict[str, Any] = {
            "name": c["name"],
            "conf_path": str(conf_path),
            "params": dict(c["params"]),
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

            item["run_id"] = run_id
            item["ok"] = bool(metrics.get("status") == "success")
            item["status"] = metrics.get("status")
            item["ops_gate_ok"] = bool(gate_payload.get("ok"))
            item["ops_gate_violations"] = gate_payload.get("violations", [])

            mm = ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test") or {}
            bt = ((metrics.get("backtest") or {}).get("summary") or {})
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

    ranked = sorted(results, key=lambda x: float(x.get("score", -1e9)), reverse=True)
    gated_ranked = [r for r in ranked if bool(r.get("ops_gate_ok"))]
    selected = gated_ranked[0] if gated_ranked else (ranked[0] if ranked else None)

    payload = {
        "base_conf": str(base_conf_path),
        "policy_path": str(policy_path),
        "selected_end_date": selected_end_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ranked": ranked,
        "selected": selected,
    }
    out_json = out_dir / "model_search_prod_results.json"
    out_md = out_dir / f"model_search_prod_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    lines = [
        f"# Model Search (Production) - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        f"- base_conf: `{base_conf_path}`",
        f"- policy: `{policy_path}`",
        f"- selected_end_date: `{selected_end_date}`",
        f"- trials: `{len(results)}`",
        "",
        "| rank | name | score | rank_ic_test_mean | sharpe | total_return | max_drawdown | ops_gate | run_id |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        gate_txt = "pass" if bool(r.get("ops_gate_ok")) else "fail"
        lines.append(
            f"| {i} | {r.get('name')} | {r.get('score')} | {r.get('rank_ic_test_mean')} | {r.get('sharpe')} | {r.get('total_return')} | {r.get('max_drawdown')} | {gate_txt} | {r.get('run_id','-')} |"
        )
    if selected:
        lines += [
            "",
            "## Selected",
            "",
            f"- name: `{selected.get('name')}`",
            f"- run_id: `{selected.get('run_id')}`",
            f"- score: `{selected.get('score')}`",
            f"- params: `{json.dumps(selected.get('params', {}), ensure_ascii=True)}`",
        ]
    out_md.write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "results_json": str(out_json),
                "results_md": str(out_md),
                "selected": selected,
                "top": ranked[:3],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
