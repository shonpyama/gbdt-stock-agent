from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.operations import collect_ops_status, write_ops_snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True))


def test_collect_ops_status_success_and_snapshot(tmp_path: Path) -> None:
    run_id = "r_ok"
    now = datetime.now(timezone.utc)

    _write_json(
        tmp_path / "state" / "last_run_state.json",
        {
            "run_id": run_id,
            "stage": "stage_80_report_ready",
            "updated_at": now.isoformat(),
        },
    )
    _write_json(
        tmp_path / "artifacts" / "runs" / run_id / "metrics.json",
        {
            "status": "success",
            "errors": [],
            "historical_errors": [],
            "training_info": {"gbdt": {"accelerator": "gpu"}},
        },
    )
    (tmp_path / "artifacts" / "runs" / run_id / "report.md").write_text("# report")

    payload = collect_ops_status(project_dir=tmp_path, max_age_hours=24, require_gpu=True)
    assert payload["ok"] is True
    assert payload["checks"]["status_success"] is True
    assert payload["checks"]["stage_80_ready"] is True
    assert payload["checks"]["gpu_accelerator"] is True

    out = write_ops_snapshot(project_dir=tmp_path, payload=payload)
    assert out.exists()
    assert (tmp_path / "logs" / "ops" / out.name).exists()


def test_collect_ops_status_detects_stale_and_failures(tmp_path: Path) -> None:
    run_id = "r_bad"
    stale = datetime.now(timezone.utc) - timedelta(hours=200)

    _write_json(
        tmp_path / "state" / "last_run_state.json",
        {
            "run_id": run_id,
            "stage": "stage_70_backtest_ready",
            "updated_at": stale.isoformat(),
        },
    )
    _write_json(
        tmp_path / "artifacts" / "runs" / run_id / "metrics.json",
        {
            "status": "error",
            "errors": [{"type": "RuntimeError"}],
            "training_info": {"gbdt": {"accelerator": "cpu"}},
        },
    )

    payload = collect_ops_status(project_dir=tmp_path, max_age_hours=72, require_gpu=False)
    assert payload["ok"] is False
    assert payload["checks"]["updated_recent"] is False
    assert payload["checks"]["stage_80_ready"] is False
    assert payload["checks"]["status_success"] is False
    assert payload["checks"]["active_errors_empty"] is False
