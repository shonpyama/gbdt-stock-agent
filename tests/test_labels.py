from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.labels import build_future_return_labels


def test_label_alignment_and_target_column_name() -> None:
    dates = pd.bdate_range("2024-01-01", periods=40).date
    rows = []
    px = 100.0
    for d in dates:
        rows.append({"date": d, "symbol": "AAA", "close": px, "adj_close": px})
        px *= 1.001
    df = pd.DataFrame(rows)

    out = build_future_return_labels(df, adjusted_flag=True, horizon_trading_days=20)
    assert "future_return_20d" in out.labels.columns
    assert out.summary["target_col"] == "future_return_20d"
    assert (out.labels["label_date"] > out.labels["decision_date"]).all()
