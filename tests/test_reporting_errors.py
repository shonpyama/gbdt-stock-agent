from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.reporting import render_report_md


def _base_kwargs() -> dict:
    return {
        "cfg": {"run": {"seed": 42}, "data": {}, "models": {}, "split": {}},
        "run_id": "r1",
        "conf_hash": "abcdef1234567890",
        "dataset_id": None,
        "feature_store_id": None,
        "code_hash": None,
        "universe_info": None,
        "split_info": None,
        "validation": None,
        "leakage": None,
        "training_info": None,
        "model_metrics": None,
        "chosen_model": None,
        "backtest_summary": None,
    }


def test_report_treats_legacy_errors_as_historical_on_success() -> None:
    body = render_report_md(
        **_base_kwargs(),
        status="SUCCESS",
        errors=[{"type": "RuntimeError"}],
        historical_errors=None,
    )
    assert "- active_error_count: `0`" in body
    assert "- historical_error_count: `1`" in body


def test_report_keeps_active_errors_on_failure() -> None:
    body = render_report_md(
        **_base_kwargs(),
        status="ERROR",
        errors=[{"type": "RuntimeError"}],
        historical_errors=[],
    )
    assert "- active_error_count: `1`" in body
    assert "- historical_error_count: `0`" in body
