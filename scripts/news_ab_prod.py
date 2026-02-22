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


def _resolve_conf_path(base: str) -> Path:
    p = Path(base)
    if p.is_absolute():
        return p
    return PROJECT_DIR / p


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B compare include_news true vs false on production setup.")
    parser.add_argument("--base-conf", default="conf/default.yaml")
    parser.add_argument("--end-date", default=str(os.environ.get("NEWS_AB_END_DATE", "")))
    args = parser.parse_args()

    project_dir = PROJECT_DIR
    paths = ProjectPaths.from_project_dir(project_dir)
    base_conf_path = _resolve_conf_path(args.base_conf)
    base_cfg = yaml.safe_load(base_conf_path.read_text())

    selected_end_date = args.end_date.strip() or str((date.today() - timedelta(days=1)).isoformat())

    policy_path = project_dir / "conf" / "ops_policy.yaml"
    policy = load_ops_policy(policy_path)
    max_age_hours = float(policy.get("max_age_hours", 72.0))
    require_gpu = bool(policy.get("require_gpu", False))

    out_dir = project_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = project_dir / "conf" / "experiments"
    conf_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        {"name": "news_on", "include_news": True},
        {"name": "news_off", "include_news": False},
    ]
    results: List[Dict[str, Any]] = []
    for c in candidates:
        cfg = deepcopy(base_cfg)
        cfg.setdefault("data", {})["end_date"] = selected_end_date
        cfg.setdefault("data", {})["include_news"] = bool(c["include_news"])
        cfg.setdefault("run", {})["log_level"] = "WARNING"
        conf_path = conf_dir / f"news_ab_prod_{c['name']}_{selected_end_date.replace('-', '')}.yaml"
        conf_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        item: Dict[str, Any] = {
            "name": c["name"],
            "include_news": bool(c["include_news"]),
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

    ranked = sorted(
        results,
        key=lambda x: (
            1 if bool(x.get("ops_gate_ok")) else 0,
            _safe_float(x.get("score")),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked else None

    payload = {
        "base_conf": str(base_conf_path),
        "policy_path": str(policy_path),
        "selected_end_date": selected_end_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ranked": ranked,
        "selected": selected,
    }
    out_json = out_dir / "news_ab_prod_results.json"
    out_md = out_dir / f"news_ab_prod_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    lines = [
        f"# News A/B (Production) - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        f"- base_conf: `{base_conf_path}`",
        f"- policy: `{policy_path}`",
        f"- selected_end_date: `{selected_end_date}`",
        "",
        "| rank | name | include_news | score | rank_ic_test_mean | sharpe | total_return | max_drawdown | ops_gate | run_id |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        gate_txt = "pass" if bool(r.get("ops_gate_ok")) else "fail"
        lines.append(
            f"| {i} | {r.get('name')} | {r.get('include_news')} | {r.get('score')} | {r.get('rank_ic_test_mean')} | {r.get('sharpe')} | {r.get('total_return')} | {r.get('max_drawdown')} | {gate_txt} | {r.get('run_id','-')} |"
        )
    if selected:
        lines += [
            "",
            "## Selected",
            "",
            f"- name: `{selected.get('name')}`",
            f"- include_news: `{selected.get('include_news')}`",
            f"- score: `{selected.get('score')}`",
        ]
    out_md.write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "results_json": str(out_json),
                "results_md": str(out_md),
                "selected": selected,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
