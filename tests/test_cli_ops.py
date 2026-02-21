from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.cli import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True))


def test_cli_ops_status_success(tmp_path: Path) -> None:
    run_id = "r_cli_ok"
    _write_json(
        tmp_path / "state" / "last_run_state.json",
        {
            "run_id": run_id,
            "stage": "stage_80_report_ready",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(
        tmp_path / "artifacts" / "runs" / run_id / "metrics.json",
        {
            "status": "success",
            "errors": [],
            "training_info": {"gbdt": {"accelerator": "gpu"}},
        },
    )
    (tmp_path / "artifacts" / "runs" / run_id / "report.md").write_text("# report")

    prev = Path.cwd()
    os.chdir(tmp_path)
    try:
        rc = main(["ops-status", "--max-age-hours", "24", "--require-gpu"])
    finally:
        os.chdir(prev)
    assert rc == 0


def test_cli_ops_snapshot_failure_exit_code(tmp_path: Path) -> None:
    run_id = "r_cli_fail"
    _write_json(
        tmp_path / "state" / "last_run_state.json",
        {
            "run_id": run_id,
            "stage": "stage_70_backtest_ready",
            "updated_at": datetime.now(timezone.utc).isoformat(),
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

    prev = Path.cwd()
    os.chdir(tmp_path)
    try:
        rc = main(["ops-snapshot", "--max-age-hours", "24", "--require-gpu"])
    finally:
        os.chdir(prev)
    assert rc == 1
    assert sorted((tmp_path / "reports").glob("ops_snapshot_*.md"))
    assert sorted((tmp_path / "logs" / "ops").glob("ops_snapshot_*.md"))
