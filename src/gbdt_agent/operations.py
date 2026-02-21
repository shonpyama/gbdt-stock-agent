from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import deep_merge, load_config
from .paths import ProjectPaths

DEFAULT_OPS_POLICY: Dict[str, Any] = {
    "max_age_hours": 72.0,
    "require_gpu": True,
    "require_validation_passed": True,
    "require_leakage_passed": True,
    "thresholds": {
        "min_sharpe": 0.5,
        "min_total_return": -0.25,
        # Drawdown is typically <= 0, so higher is better (e.g. -0.20 > -0.40).
        "min_max_drawdown": -0.45,
        "max_avg_cost": 0.03,
    },
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_iso_utc(value: Any) -> Optional[datetime]:
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


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def load_ops_policy(policy_path: Optional[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = json.loads(json.dumps(DEFAULT_OPS_POLICY))
    if policy_path is None:
        return merged
    if not policy_path.exists():
        return merged
    cfg = load_config(policy_path)
    if not isinstance(cfg, dict):
        return merged
    deep_merge(merged, cfg)
    return merged


def collect_ops_status(
    *,
    project_dir: Path,
    run_id: Optional[str] = None,
    max_age_hours: float = 72.0,
    require_gpu: bool = False,
) -> Dict[str, Any]:
    paths = ProjectPaths.from_project_dir(project_dir)
    state = _read_json(paths.state_dir / "last_run_state.json")
    selected_run_id = run_id or (state.get("run_id") if isinstance(state, dict) else None)

    metrics: Dict[str, Any] = {}
    metrics_path: Optional[Path] = None
    report_path: Optional[Path] = None
    manifest: Dict[str, Any] = {}
    manifest_path: Optional[Path] = None
    if isinstance(selected_run_id, str) and selected_run_id:
        run_dir = paths.run_dir(selected_run_id)
        metrics_path = run_dir / "metrics.json"
        report_path = run_dir / "report.md"
        manifest_path = run_dir / "artifact_manifest.json"
        metrics = _read_json(metrics_path)
        manifest = _read_json(manifest_path)

    # last_run_state.json reflects only the latest run, so only trust it for matching run_id.
    state_for_run = (
        state
        if (
            isinstance(selected_run_id, str)
            and selected_run_id
            and isinstance(state, dict)
            and str(state.get("run_id", "")) == selected_run_id
        )
        else {}
    )

    now = datetime.now(timezone.utc)
    updated_at_dt = (
        _parse_iso_utc((state_for_run or {}).get("updated_at"))
        or _parse_iso_utc((metrics or {}).get("updated_at"))
        or _parse_iso_utc((manifest or {}).get("generated_at"))
    )
    if updated_at_dt is None and metrics_path and metrics_path.exists():
        updated_at_dt = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=timezone.utc)
    age_hours: Optional[float] = None
    if updated_at_dt is not None:
        age_hours = max(0.0, (now - updated_at_dt).total_seconds() / 3600.0)

    stage = str((state_for_run or {}).get("stage", "") or (manifest or {}).get("stage", ""))
    status = str((metrics or {}).get("status", ""))
    active_errors = (metrics or {}).get("errors")
    if not isinstance(active_errors, list):
        active_errors = []
    accelerator = (((metrics or {}).get("training_info") or {}).get("gbdt") or {}).get("accelerator")
    gpu_accelerator = str(accelerator).lower() == "gpu"
    backtest_summary = ((metrics or {}).get("backtest") or {}).get("summary") or {}
    validation = (metrics or {}).get("validation") or {}
    leakage = (metrics or {}).get("leakage") or {}

    checks: Dict[str, bool] = {
        "state_exists": bool((paths.state_dir / "last_run_state.json").exists()),
        "run_id_present": bool(selected_run_id),
        "metrics_exists": bool(metrics_path and metrics_path.exists()),
        "report_exists": bool(report_path and report_path.exists()),
        "stage_80_ready": stage == "stage_80_report_ready",
        "status_success": status == "success",
        "active_errors_empty": len(active_errors) == 0,
        "updated_recent": (age_hours is not None and age_hours <= float(max_age_hours)),
    }
    checks["gpu_accelerator"] = gpu_accelerator if require_gpu else True

    optional_signals = {
        "latest_transition_report_exists": bool(sorted(paths.reports_dir.glob("pre_colab_transition_*.md"))),
        "ops_snapshot_exists": bool(sorted((paths.logs_dir / "ops").glob("ops_snapshot_*.md"))),
    }

    return {
        "ok": all(checks.values()),
        "run_id": selected_run_id,
        "stage": stage or None,
        "status": status or None,
        "updated_at": updated_at_dt.isoformat() if updated_at_dt else None,
        "age_hours": round(age_hours, 3) if age_hours is not None else None,
        "max_age_hours": float(max_age_hours),
        "require_gpu": bool(require_gpu),
        "accelerator": accelerator,
        "last_data_date": (state_for_run or {}).get("last_data_date") or (metrics or {}).get("last_data_date"),
        "validation_passed": validation.get("passed"),
        "leakage_passed": leakage.get("passed"),
        "backtest_summary": {
            "sharpe": _safe_float(backtest_summary.get("sharpe")),
            "max_drawdown": _safe_float(backtest_summary.get("max_drawdown")),
            "total_return": _safe_float(backtest_summary.get("total_return")),
            "avg_turnover": _safe_float(backtest_summary.get("avg_turnover")),
            "avg_cost": _safe_float(backtest_summary.get("avg_cost")),
            "days": _safe_int(backtest_summary.get("days")),
        },
        "active_error_count": len(active_errors),
        "historical_error_count": len((metrics or {}).get("historical_errors") or []),
        "checks": checks,
        "signals": optional_signals,
        "paths": {
            "project_dir": str(paths.project_dir),
            "state": str(paths.state_dir / "last_run_state.json"),
            "metrics": str(metrics_path) if metrics_path else None,
            "report": str(report_path) if report_path else None,
            "manifest": str(manifest_path) if manifest_path else None,
        },
    }


def evaluate_ops_gate(status_payload: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    thresholds = policy.get("thresholds", {}) if isinstance(policy.get("thresholds", {}), dict) else {}
    violations: List[str] = []
    checks: Dict[str, bool] = {}

    status_checks = status_payload.get("checks") if isinstance(status_payload.get("checks"), dict) else {}
    for base_key in [
        "state_exists",
        "run_id_present",
        "metrics_exists",
        "report_exists",
        "stage_80_ready",
        "status_success",
        "active_errors_empty",
        "updated_recent",
        "gpu_accelerator",
    ]:
        checks[base_key] = bool(status_checks.get(base_key))
        if not checks[base_key]:
            violations.append(f"base_check_failed:{base_key}")

    require_validation = bool(policy.get("require_validation_passed", True))
    require_leakage = bool(policy.get("require_leakage_passed", True))
    validation_passed = bool(status_payload.get("validation_passed"))
    leakage_passed = bool(status_payload.get("leakage_passed"))

    checks["validation_passed"] = (validation_passed if require_validation else True)
    checks["leakage_passed"] = (leakage_passed if require_leakage else True)
    if require_validation and not validation_passed:
        violations.append("gate_failed:validation_passed")
    if require_leakage and not leakage_passed:
        violations.append("gate_failed:leakage_passed")

    bt = status_payload.get("backtest_summary") if isinstance(status_payload.get("backtest_summary"), dict) else {}
    sharpe = _safe_float(bt.get("sharpe"))
    total_return = _safe_float(bt.get("total_return"))
    max_drawdown = _safe_float(bt.get("max_drawdown"))
    avg_cost = _safe_float(bt.get("avg_cost"))

    min_sharpe = _safe_float(thresholds.get("min_sharpe"))
    min_total_return = _safe_float(thresholds.get("min_total_return"))
    min_max_drawdown = _safe_float(thresholds.get("min_max_drawdown"))
    max_avg_cost = _safe_float(thresholds.get("max_avg_cost"))

    checks["min_sharpe"] = (True if min_sharpe is None else (sharpe is not None and sharpe >= min_sharpe))
    checks["min_total_return"] = (
        True if min_total_return is None else (total_return is not None and total_return >= min_total_return)
    )
    checks["min_max_drawdown"] = (
        True if min_max_drawdown is None else (max_drawdown is not None and max_drawdown >= min_max_drawdown)
    )
    checks["max_avg_cost"] = (True if max_avg_cost is None else (avg_cost is not None and avg_cost <= max_avg_cost))

    if not checks["min_sharpe"]:
        violations.append(f"threshold_failed:min_sharpe value={sharpe} required>={min_sharpe}")
    if not checks["min_total_return"]:
        violations.append(f"threshold_failed:min_total_return value={total_return} required>={min_total_return}")
    if not checks["min_max_drawdown"]:
        violations.append(f"threshold_failed:min_max_drawdown value={max_drawdown} required>={min_max_drawdown}")
    if not checks["max_avg_cost"]:
        violations.append(f"threshold_failed:max_avg_cost value={avg_cost} required<={max_avg_cost}")

    ok = all(checks.values())
    return {
        "ok": ok,
        "run_id": status_payload.get("run_id"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "checks": checks,
        "violations": violations,
        "status_payload": status_payload,
    }


def write_ops_incident(*, project_dir: Path, payload: Dict[str, Any]) -> Path:
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%SZ")
    paths = ProjectPaths.from_project_dir(project_dir)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    out = paths.reports_dir / f"ops_incident_{ts}.md"
    lines = [
        "# Ops Incident",
        "",
        f"- generated_at: `{now.isoformat()}`",
        f"- ok: `{payload.get('ok')}`",
        f"- run_id: `{payload.get('run_id')}`",
        "",
        "## Violations",
        "",
    ]
    violations = payload.get("violations") if isinstance(payload.get("violations"), list) else []
    if violations:
        lines.extend([f"- {v}" for v in violations])
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Raw Payload",
        "",
        "```json",
        json.dumps(payload, indent=2, ensure_ascii=True),
        "```",
        "",
    ]
    out.write_text("\n".join(lines))

    ops_dir = paths.logs_dir / "ops"
    ops_dir.mkdir(parents=True, exist_ok=True)
    mirror = ops_dir / out.name
    mirror.write_text(out.read_text())
    return out


def write_ops_snapshot(*, project_dir: Path, payload: Dict[str, Any]) -> Path:
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%SZ")
    paths = ProjectPaths.from_project_dir(project_dir)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    out = paths.reports_dir / f"ops_snapshot_{ts}.md"
    lines = [
        "# Ops Snapshot",
        "",
        f"- generated_at: `{now.isoformat()}`",
        f"- ok: `{payload.get('ok')}`",
        f"- run_id: `{payload.get('run_id')}`",
        f"- stage: `{payload.get('stage')}`",
        f"- status: `{payload.get('status')}`",
        f"- accelerator: `{payload.get('accelerator')}`",
        f"- active_error_count: `{payload.get('active_error_count')}`",
        f"- historical_error_count: `{payload.get('historical_error_count')}`",
        f"- age_hours: `{payload.get('age_hours')}`",
        f"- max_age_hours: `{payload.get('max_age_hours')}`",
        "",
        "## Checks",
        "",
    ]
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    for k in sorted(checks.keys()):
        lines.append(f"- {k}: `{checks[k]}`")
    lines += [
        "",
        "## Raw Payload",
        "",
        "```json",
        json.dumps(payload, indent=2, ensure_ascii=True),
        "```",
        "",
    ]
    out.write_text("\n".join(lines))

    ops_dir = paths.logs_dir / "ops"
    ops_dir.mkdir(parents=True, exist_ok=True)
    mirror = ops_dir / out.name
    mirror.write_text(out.read_text())
    return out
