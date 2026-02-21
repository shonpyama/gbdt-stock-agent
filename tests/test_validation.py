from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.validation import validate_raw_prices


def _base_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=20).date
    rows = []
    for symbol in ["AAA", "BBB"]:
        px = 100.0
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "symbol": symbol,
                    "open": px,
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "adj_close": px,
                    "volume": 1000,
                }
            )
            px *= 1.001
    return pd.DataFrame(rows)


def test_validation_passes_for_clean_data() -> None:
    df = _base_prices()
    res = validate_raw_prices(
        df,
        adjusted_flag=True,
        thresholds={
            "missing_ratio_max": 0.05,
            "anomaly_row_ratio_max": 0.02,
            "duplicates_ratio_max": 0.0,
            "min_obs_per_symbol": 5,
            "min_symbols_after_filter": 1,
            "adjclose_missing_ratio_max": 0.05,
            "volume_zero_streak_len": 3,
            "volume_zero_streak_ratio_max": 0.2,
        },
    )
    assert res.passed is True
    assert res.summary["fail_reasons"] == []


def test_validation_fails_on_zero_volume_streak() -> None:
    df = _base_prices()
    m = (df["symbol"] == "AAA") & (pd.to_datetime(df["date"]) >= pd.Timestamp("2024-01-08")) & (
        pd.to_datetime(df["date"]) <= pd.Timestamp("2024-01-12")
    )
    df.loc[m, "volume"] = 0
    res = validate_raw_prices(
        df,
        adjusted_flag=True,
        thresholds={
            "missing_ratio_max": 0.05,
            "anomaly_row_ratio_max": 0.05,
            "duplicates_ratio_max": 0.0,
            "min_obs_per_symbol": 5,
            "min_symbols_after_filter": 1,
            "adjclose_missing_ratio_max": 0.05,
            "volume_zero_streak_len": 3,
            "volume_zero_streak_ratio_max": 0.01,
        },
    )
    assert res.passed is False
    assert any("zero_volume_streak_violation_ratio" in x for x in res.summary["fail_reasons"])
