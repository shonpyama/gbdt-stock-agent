from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import ProjectPaths


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
    if isinstance(selected_run_id, str) and selected_run_id:
        run_dir = paths.run_dir(selected_run_id)
        metrics_path = run_dir / "metrics.json"
        report_path = run_dir / "report.md"
        metrics = _read_json(metrics_path)

    now = datetime.now(timezone.utc)
    updated_at_dt = _parse_iso_utc((state or {}).get("updated_at")) or _parse_iso_utc((metrics or {}).get("updated_at"))
    age_hours: Optional[float] = None
    if updated_at_dt is not None:
        age_hours = max(0.0, (now - updated_at_dt).total_seconds() / 3600.0)

    stage = str((state or {}).get("stage", ""))
    status = str((metrics or {}).get("status", ""))
    active_errors = (metrics or {}).get("errors")
    if not isinstance(active_errors, list):
        active_errors = []
    accelerator = (((metrics or {}).get("training_info") or {}).get("gbdt") or {}).get("accelerator")
    gpu_accelerator = str(accelerator).lower() == "gpu"

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
        "active_error_count": len(active_errors),
        "historical_error_count": len((metrics or {}).get("historical_errors") or []),
        "checks": checks,
        "signals": optional_signals,
        "paths": {
            "project_dir": str(paths.project_dir),
            "state": str(paths.state_dir / "last_run_state.json"),
            "metrics": str(metrics_path) if metrics_path else None,
            "report": str(report_path) if report_path else None,
        },
    }


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
