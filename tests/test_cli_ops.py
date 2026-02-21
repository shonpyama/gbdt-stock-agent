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
            "validation": {"passed": True},
            "leakage": {"passed": True},
            "backtest": {"summary": {"sharpe": 0.8, "max_drawdown": -0.2, "total_return": 0.1, "avg_cost": 0.01}},
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


def test_cli_ops_gate_writes_incident_on_failure(tmp_path: Path) -> None:
    run_id = "r_cli_gate_fail"
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
            "validation": {"passed": True},
            "leakage": {"passed": True},
            "backtest": {"summary": {"sharpe": 0.1, "max_drawdown": -0.7, "total_return": -0.5, "avg_cost": 0.2}},
        },
    )
    (tmp_path / "artifacts" / "runs" / run_id / "report.md").write_text("# report")
    (tmp_path / "conf").mkdir(parents=True, exist_ok=True)
    (tmp_path / "conf" / "ops_policy.yaml").write_text(
        "max_age_hours: 24\nrequire_gpu: true\nrequire_validation_passed: true\nrequire_leakage_passed: true\nthresholds:\n  min_sharpe: 0.5\n  min_total_return: 0.0\n  min_max_drawdown: -0.3\n  max_avg_cost: 0.03\n"
    )

    prev = Path.cwd()
    os.chdir(tmp_path)
    try:
        rc = main(["ops-gate", "--policy", "conf/ops_policy.yaml"])
    finally:
        os.chdir(prev)
    assert rc == 1
    assert sorted((tmp_path / "reports").glob("ops_incident_*.md"))
    assert sorted((tmp_path / "logs" / "ops").glob("ops_incident_*.md"))
