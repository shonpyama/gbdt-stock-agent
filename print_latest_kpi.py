#!/usr/bin/env python3
import json
from pathlib import Path


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    manifest = load_json("outputs/reports/latest_run_manifest.json")
    gate = load_json("outputs/reports/latest_gate.json")
    detail = load_json("outputs/reports/latest_detailed_report.json")
    attempts = load_json("outputs/reports/latest_improve_attempts.json")

    wf = detail.get("walk_forward", {})
    rules = detail.get("exit_rules_comparison", {})
    cfg = detail.get("meta", {}).get("run_config", {})
    best_rule = None
    best_sharpe = float("-inf")
    for name, r in rules.items():
        if not bool(r.get("eligible", True)):
            continue
        s = float(r.get("sharpe_net", 0.0))
        if s > best_sharpe:
            best_sharpe = s
            best_rule = name

    print("Latest GBDT KPI")
    print(f"  generated_at: {manifest.get('generated_at', 'n/a')}")
    print(f"  symbols: {manifest.get('symbols', 'n/a')}")
    print(f"  gate: {gate.get('status', manifest.get('gate_status', 'n/a'))}")
    print(f"  gate_reasons: {', '.join(gate.get('reasons', [])) or 'none'}")
    print(f"  wf_avg_acc: {float(wf.get('avg_accuracy', 0.0)):.4f}")
    print(f"  wf_avg_auc: {float(wf.get('avg_auc', 0.0)):.4f}")
    print(f"  best_rule: {best_rule or 'none'}")
    print(f"  best_sharpe: {best_sharpe if best_rule else 0.0:.4f}")
    print(
        "  device_holdout/live: "
        f"{cfg.get('train_device_holdout', 'n/a')}/{cfg.get('train_device_live', 'n/a')}"
    )
    print(f"  log_path: {manifest.get('artifacts', {}).get('log_path', 'n/a')}")

    arr = attempts.get("attempts", [])
    if arr:
        last = arr[-1]
        print(
            "  last_attempt: "
            f"{last.get('label', 'n/a')} ({last.get('gate_status', 'n/a')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
