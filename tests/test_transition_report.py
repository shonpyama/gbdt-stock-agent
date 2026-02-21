from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.transition import generate_transition_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True))


def test_transition_report_hides_historical_errors_after_success(tmp_path: Path) -> None:
    run_id = "r1"
    _write_json(
        tmp_path / "artifacts" / "runs" / run_id / "metrics.json",
        {
            "status": "success",
            "errors": [{"type": "RuntimeError", "message": "old failure"}],
            "leakage": {"passed": True},
            "backtest": {"chosen_model": "gbdt", "summary": {}},
        },
    )
    _write_json(tmp_path / "artifacts" / "runs" / run_id / "artifact_manifest.json", {"run_id": run_id})
    _write_json(tmp_path / "state" / "last_run_state.json", {"stage": "stage_80_report_ready"})

    out = generate_transition_report(project_dir=tmp_path, run_id=run_id, target="colab")
    body = out.read_text()
    assert "- error_count: `0`" in body
    assert "- historical_error_count: `1`" in body


def test_transition_report_shows_active_errors_when_not_success(tmp_path: Path) -> None:
    run_id = "r2"
    _write_json(
        tmp_path / "artifacts" / "runs" / run_id / "metrics.json",
        {
            "status": "error",
            "errors": [{"type": "RuntimeError", "message": "active failure"}],
            "leakage": {"passed": False},
            "backtest": {"chosen_model": "gbdt", "summary": {}},
        },
    )
    _write_json(tmp_path / "artifacts" / "runs" / run_id / "artifact_manifest.json", {"run_id": run_id})
    _write_json(tmp_path / "state" / "last_run_state.json", {"stage": "stage_50_models_trained"})

    out = generate_transition_report(project_dir=tmp_path, run_id=run_id, target="colab")
    body = out.read_text()
    assert "- error_count: `1`" in body
    assert "- historical_error_count:" not in body
    assert "active failure" in body


def test_transition_report_uses_historical_errors_field(tmp_path: Path) -> None:
    run_id = "r3"
    _write_json(
        tmp_path / "artifacts" / "runs" / run_id / "metrics.json",
        {
            "status": "success",
            "errors": [],
            "historical_errors": [{"type": "RuntimeError", "message": "archived"}],
            "leakage": {"passed": True},
            "backtest": {"chosen_model": "gbdt", "summary": {}},
            "training_info": {"gbdt": {"accelerator": "gpu", "gpu_attempted": True}},
        },
    )
    _write_json(tmp_path / "artifacts" / "runs" / run_id / "artifact_manifest.json", {"run_id": run_id})
    _write_json(tmp_path / "state" / "last_run_state.json", {"stage": "stage_80_report_ready"})

    out = generate_transition_report(project_dir=tmp_path, run_id=run_id, target="colab")
    body = out.read_text()
    assert "- error_count: `0`" in body
    assert "- historical_error_count: `1`" in body
    assert "- train_accelerator: `gpu`" in body
