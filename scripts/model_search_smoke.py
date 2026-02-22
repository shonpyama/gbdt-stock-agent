#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    # Rank quality first, then risk-adjusted performance.
    return (rank_ic * 100.0) + (sharpe * 0.5) + (total_ret * 10.0) + (max_dd * 2.0)


def main() -> int:
    project_dir = PROJECT_DIR
    paths = ProjectPaths.from_project_dir(project_dir)
    base_conf_path = project_dir / "conf" / "smoke_local.yaml"
    base_cfg = yaml.safe_load(base_conf_path.read_text())

    candidates: List[Dict[str, Any]] = [
        {
            "name": "baseline_smoke_local",
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
            },
        },
        {
            "name": "prod_default_auto",
            "params": {},
        },
        {
            "name": "l2_regularized_63",
            "params": {
                "n_estimators": 3500,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "min_child_samples": 80,
                "reg_lambda": 2.0,
                "reg_alpha": 0.2,
            },
        },
        {
            "name": "low_lr_95",
            "params": {
                "n_estimators": 5000,
                "learning_rate": 0.02,
                "num_leaves": 95,
                "subsample": 0.9,
                "colsample_bytree": 0.7,
                "min_child_samples": 70,
                "reg_lambda": 1.0,
            },
        },
        {
            "name": "higher_leaf_127",
            "params": {
                "n_estimators": 2500,
                "learning_rate": 0.04,
                "num_leaves": 127,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 100,
                "reg_lambda": 1.0,
            },
        },
    ]

    out_dir = project_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = project_dir / "conf" / "experiments"
    conf_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for idx, c in enumerate(candidates, start=1):
        cfg = deepcopy(base_cfg)
        cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = dict(c["params"])
        conf_path = conf_dir / f"smoke_model_{idx:02d}_{c['name']}.yaml"
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
            item["run_id"] = run_id
            item["ok"] = bool(metrics.get("status") == "success")
            item["status"] = metrics.get("status")
            mm = ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test") or {}
            bt = ((metrics.get("backtest") or {}).get("summary") or {})
            item["rank_ic_test_mean"] = ((mm.get("rank_ic") or {}).get("mean"))
            item["ic_test_mean"] = ((mm.get("ic") or {}).get("mean"))
            item["sharpe"] = bt.get("sharpe")
            item["total_return"] = bt.get("total_return")
            item["max_drawdown"] = bt.get("max_drawdown")
            item["score"] = _score(metrics)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    ranked = sorted(results, key=lambda x: float(x.get("score", -1e9)), reverse=True)
    payload = {"base_conf": str(base_conf_path), "ranked": ranked}
    out_json = out_dir / "model_search_smoke_results.json"
    out_md = out_dir / "model_search_smoke_results.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    lines = [
        "# Model Search (Smoke)",
        "",
        f"- base_conf: `{base_conf_path}`",
        f"- trials: `{len(results)}`",
        "",
        "| rank | name | score | rank_ic_test_mean | sharpe | total_return | max_drawdown | run_id |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {r.get('name')} | {r.get('score')} | {r.get('rank_ic_test_mean')} | {r.get('sharpe')} | {r.get('total_return')} | {r.get('max_drawdown')} | {r.get('run_id','-')} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "results_json": str(out_json),
                "results_md": str(out_md),
                "top": ranked[:3],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
