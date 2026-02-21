from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.state import append_state_error, last_success_stage, load_last_state, save_last_state


def test_last_success_stage_for_failed_stage() -> None:
    assert last_success_stage("stage_20_validation_failed") == "stage_10_data_ready"
    assert last_success_stage("stage_40_split_leakcheck_failed") == "stage_30_features_ready"


def test_state_save_load_normalizes_defaults(tmp_path: Path) -> None:
    st = {
        "run_id": "r1",
        "stage": "stage_10_data_ready",
        "conf_hash": "abc",
        "dataset_id": "d1",
    }
    save_last_state(tmp_path, st)
    loaded = load_last_state(tmp_path)
    assert loaded is not None
    assert loaded["run_id"] == "r1"
    assert loaded["feature_store_id"] is None
    assert isinstance(loaded["model_ckpt_paths"], dict)
    assert loaded["updated_at"] is not None


def test_append_state_error_keeps_history() -> None:
    st = {"run_id": "r1", "stage": "stage_20_validation_failed", "errors": []}
    out = append_state_error(st, stage="stage_20_validation_failed", error="oops")
    assert len(out["errors"]) == 1
    assert out["errors"][0]["stage"] == "stage_20_validation_failed"
