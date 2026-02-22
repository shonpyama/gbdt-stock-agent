#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gbdt_agent.operations import collect_ops_status, evaluate_ops_gate, load_ops_policy


def _parse_iso_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _check_stability_payload(
    payload: Dict[str, Any],
    *,
    required_periods: int,
    max_age_hours: float,
) -> Tuple[bool, Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    selected = payload.get("selected") if isinstance(payload.get("selected"), dict) else {}
    generated_at = _parse_iso_utc(payload.get("generated_at"))
    age_hours = (now - generated_at).total_seconds() / 3600.0 if generated_at else None
    periods = int(selected.get("periods") or 0)
    all_gate = bool(selected.get("all_periods_gate_pass"))
    has_selected = bool(selected)
    fresh = (age_hours is not None and age_hours <= float(max_age_hours))
    ok = has_selected and all_gate and periods >= int(required_periods) and fresh
    return ok, {
        "has_selected": has_selected,
        "all_periods_gate_pass": all_gate,
        "periods": periods,
        "required_periods": int(required_periods),
        "generated_at": generated_at.isoformat() if generated_at else None,
        "age_hours": round(age_hours, 3) if age_hours is not None else None,
        "max_age_hours": float(max_age_hours),
        "fresh": fresh,
        "selected": selected,
    }


def _check_default_alignment(default_conf: Dict[str, Any], model_selected: Dict[str, Any], feature_selected: Dict[str, Any]) -> Dict[str, Any]:
    actual_model = (
        ((default_conf.get("models") or {}).get("gbdt") or {}).get("params") or {}
    )
    actual_lookbacks = list(((default_conf.get("features") or {}).get("lookbacks") or []))
    actual_shift = int(((default_conf.get("features") or {}).get("event_safe_shift_days") or 1))

    model_expected = dict(model_selected.get("params") or {})
    feature_expected_lb = list(feature_selected.get("lookbacks") or [])
    feature_expected_shift = int(feature_selected.get("event_shift") or 1)

    return {
        "model_params_match": actual_model == model_expected,
        "feature_lookbacks_match": actual_lookbacks == feature_expected_lb,
        "feature_shift_match": actual_shift == feature_expected_shift,
        "actual_model_params": actual_model,
        "expected_model_params": model_expected,
        "actual_feature_lookbacks": actual_lookbacks,
        "expected_feature_lookbacks": feature_expected_lb,
        "actual_feature_shift": actual_shift,
        "expected_feature_shift": feature_expected_shift,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate production readiness for model/feature stability + ops gate.")
    parser.add_argument("--project-dir", default=str(PROJECT_DIR), help="Project directory")
    parser.add_argument("--default-conf", default="conf/default.yaml", help="Default config path")
    parser.add_argument("--policy", default="conf/ops_policy.yaml", help="Ops policy path")
    parser.add_argument("--model-results", default="reports/model_stability_prod_results.json", help="Model stability JSON")
    parser.add_argument("--feature-results", default="reports/feature_stability_prod_results.json", help="Feature stability JSON")
    parser.add_argument("--required-periods", type=int, default=3, help="Required periods for each stability result")
    parser.add_argument("--max-stability-age-hours", type=float, default=168.0, help="Max age for stability results")
    parser.add_argument("--run-id", default="", help="Optional run_id for ops gate")
    parser.add_argument("--out-json", default="reports/prod_readiness_check.json", help="Output JSON report path")
    parser.add_argument("--out-md", default="reports/prod_readiness_check.md", help="Output Markdown report path")
    parser.add_argument("--strict", action="store_true", help="Exit 1 when readiness is not 100")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    default_conf_path = (project_dir / args.default_conf) if not Path(args.default_conf).is_absolute() else Path(args.default_conf)
    policy_path = (project_dir / args.policy) if not Path(args.policy).is_absolute() else Path(args.policy)
    model_results_path = (project_dir / args.model_results) if not Path(args.model_results).is_absolute() else Path(args.model_results)
    feature_results_path = (project_dir / args.feature_results) if not Path(args.feature_results).is_absolute() else Path(args.feature_results)
    out_json = (project_dir / args.out_json) if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_md = (project_dir / args.out_md) if not Path(args.out_md).is_absolute() else Path(args.out_md)

    default_conf = yaml.safe_load(default_conf_path.read_text())
    model_payload = _load_json(model_results_path)
    feature_payload = _load_json(feature_results_path)

    model_ok, model_detail = _check_stability_payload(
        model_payload,
        required_periods=args.required_periods,
        max_age_hours=args.max_stability_age_hours,
    )
    feature_ok, feature_detail = _check_stability_payload(
        feature_payload,
        required_periods=args.required_periods,
        max_age_hours=args.max_stability_age_hours,
    )

    policy = load_ops_policy(policy_path)
    ops_status = collect_ops_status(
        project_dir=project_dir,
        run_id=(args.run_id or None),
        max_age_hours=float(policy.get("max_age_hours", 72.0)),
        require_gpu=bool(policy.get("require_gpu", False)),
    )
    ops_gate = evaluate_ops_gate(ops_status, policy)
    ops_ok = bool(ops_gate.get("ok"))

    alignment = _check_default_alignment(
        default_conf=default_conf,
        model_selected=model_detail.get("selected") or {},
        feature_selected=feature_detail.get("selected") or {},
    )
    align_ok = bool(
        alignment.get("model_params_match")
        and alignment.get("feature_lookbacks_match")
        and alignment.get("feature_shift_match")
    )

    checks = {
        "model_stability": model_ok,
        "feature_stability": feature_ok,
        "ops_gate": ops_ok,
        "default_conf_alignment": align_ok,
    }
    passed = sum(1 for v in checks.values() if bool(v))
    total = len(checks)
    readiness_score = round((passed / total) * 100.0, 2) if total else 0.0
    readiness_100 = (passed == total)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness_score": readiness_score,
        "readiness_100": readiness_100,
        "checks": checks,
        "model_detail": model_detail,
        "feature_detail": feature_detail,
        "ops_gate_ok": ops_ok,
        "ops_gate": ops_gate,
        "default_alignment": alignment,
        "inputs": {
            "project_dir": str(project_dir),
            "default_conf": str(default_conf_path),
            "policy": str(policy_path),
            "model_results": str(model_results_path),
            "feature_results": str(feature_results_path),
            "required_periods": int(args.required_periods),
            "max_stability_age_hours": float(args.max_stability_age_hours),
            "run_id": (args.run_id or None),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    lines = [
        "# Production Readiness Check",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- readiness_score: `{readiness_score}`",
        f"- readiness_100: `{readiness_100}`",
        "",
        "## Checks",
        "",
        f"- model_stability: `{checks['model_stability']}`",
        f"- feature_stability: `{checks['feature_stability']}`",
        f"- ops_gate: `{checks['ops_gate']}`",
        f"- default_conf_alignment: `{checks['default_conf_alignment']}`",
        "",
        "## Summary",
        "",
        f"- model_selected: `{json.dumps((model_detail.get('selected') or {}), ensure_ascii=True)}`",
        f"- feature_selected: `{json.dumps((feature_detail.get('selected') or {}), ensure_ascii=True)}`",
        f"- ops_gate_violations: `{json.dumps((ops_gate.get('violations') or []), ensure_ascii=True)}`",
        "",
    ]
    out_md.write_text("\n".join(lines))

    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "readiness_score": readiness_score, "readiness_100": readiness_100}, indent=2, ensure_ascii=True))
    if args.strict and not readiness_100:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
