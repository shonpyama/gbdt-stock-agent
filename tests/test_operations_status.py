from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.operations import collect_ops_status, evaluate_ops_gate, write_ops_incident, write_ops_snapshot


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
            "validation": {"passed": True},
            "leakage": {"passed": True},
            "backtest": {"summary": {"sharpe": 1.2, "max_drawdown": -0.12, "total_return": 0.18, "avg_cost": 0.01}},
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

    gate = evaluate_ops_gate(
        payload,
        {
            "thresholds": {"min_sharpe": 0.5, "min_total_return": 0.0, "min_max_drawdown": -0.3, "max_avg_cost": 0.03},
            "require_validation_passed": True,
            "require_leakage_passed": True,
        },
    )
    assert gate["ok"] is True


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
            "validation": {"passed": False},
            "leakage": {"passed": False},
            "backtest": {"summary": {"sharpe": -0.1, "max_drawdown": -0.8, "total_return": -0.5, "avg_cost": 0.2}},
        },
    )

    payload = collect_ops_status(project_dir=tmp_path, max_age_hours=72, require_gpu=False)
    assert payload["ok"] is False
    assert payload["checks"]["updated_recent"] is False
    assert payload["checks"]["stage_80_ready"] is False
    assert payload["checks"]["status_success"] is False
    assert payload["checks"]["active_errors_empty"] is False

    gate = evaluate_ops_gate(
        payload,
        {
            "thresholds": {"min_sharpe": 0.5, "min_total_return": 0.0, "min_max_drawdown": -0.3, "max_avg_cost": 0.03},
            "require_validation_passed": True,
            "require_leakage_passed": True,
        },
    )
    assert gate["ok"] is False
    assert any("threshold_failed:min_sharpe" in v for v in gate["violations"])
    incident = write_ops_incident(project_dir=tmp_path, payload=gate)
    assert incident.exists()
    assert (tmp_path / "logs" / "ops" / incident.name).exists()


def test_collect_ops_status_run_id_uses_run_local_stage_not_last_state(tmp_path: Path) -> None:
    run_id = "r_old"
    # last_run_state points to another run; selected run should not use this stage.
    _write_json(
        tmp_path / "state" / "last_run_state.json",
        {
            "run_id": "r_new",
            "stage": "stage_80_report_ready",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(
        tmp_path / "artifacts" / "runs" / run_id / "metrics.json",
        {"status": "success", "errors": []},
    )
    _write_json(
        tmp_path / "artifacts" / "runs" / run_id / "artifact_manifest.json",
        {"stage": "stage_50_models_trained", "generated_at": datetime.now(timezone.utc).isoformat()},
    )
    (tmp_path / "artifacts" / "runs" / run_id / "report.md").write_text("# report")

    payload = collect_ops_status(project_dir=tmp_path, run_id=run_id, max_age_hours=24, require_gpu=False)
    assert payload["stage"] == "stage_50_models_trained"
    assert payload["checks"]["stage_80_ready"] is False


def test_collect_ops_status_handles_non_numeric_days(tmp_path: Path) -> None:
    run_id = "r_days"
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
            "validation": {"passed": True},
            "leakage": {"passed": True},
            "backtest": {"summary": {"days": "nan"}},
        },
    )
    (tmp_path / "artifacts" / "runs" / run_id / "report.md").write_text("# report")
    payload = collect_ops_status(project_dir=tmp_path, run_id=run_id, max_age_hours=24, require_gpu=False)
    assert payload["backtest_summary"]["days"] is None
