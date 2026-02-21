from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.backtest import run_backtest


def _preds() -> pd.DataFrame:
    dts = pd.bdate_range("2024-01-01", periods=10).date
    rows = []
    for i, d in enumerate(dts):
        for s, score in [("AAA", 0.9), ("BBB", 0.6), ("CCC", 0.2)]:
            rows.append(
                {
                    "decision_date": d,
                    "symbol": s,
                    "y_pred": score - i * 0.01,
                    "y_true": 0.001 * (3 - (0 if s == "AAA" else 1 if s == "BBB" else 2)),
                    "adv20_dollar": 1_000_000.0,
                    "vol_20d": 0.02,
                }
            )
    return pd.DataFrame(rows)


def test_backtest_daily_runs() -> None:
    res = run_backtest(
        _preds(),
        topn=2,
        long_short=False,
        max_names=2,
        single_name_cap=0.6,
        rebalance="daily",
        commission_bps=1.0,
        slippage_base_bps=1.0,
        slippage_k_adv=0.0,
        slippage_k_vol=0.0,
    )
    assert not res.daily.empty
    assert "equity" in res.daily.columns
    assert res.summary["days"] > 0


def test_backtest_weekly_has_lower_or_equal_turnover_than_daily() -> None:
    preds = _preds()
    daily = run_backtest(
        preds,
        topn=2,
        long_short=False,
        max_names=2,
        single_name_cap=0.6,
        rebalance="daily",
        commission_bps=1.0,
        slippage_base_bps=1.0,
        slippage_k_adv=0.0,
        slippage_k_vol=0.0,
    )
    weekly = run_backtest(
        preds,
        topn=2,
        long_short=False,
        max_names=2,
        single_name_cap=0.6,
        rebalance="weekly",
        commission_bps=1.0,
        slippage_base_bps=1.0,
        slippage_k_adv=0.0,
        slippage_k_vol=0.0,
    )
    assert float(weekly.daily["turnover"].mean()) <= float(daily.daily["turnover"].mean()) + 1e-12


def test_backtest_summary_metrics_are_finite_with_cost_inputs() -> None:
    res = run_backtest(
        _preds(),
        topn=2,
        long_short=False,
        max_names=2,
        single_name_cap=0.6,
        rebalance="daily",
        commission_bps=1.0,
        slippage_base_bps=1.0,
        slippage_k_adv=5.0,
        slippage_k_vol=3.0,
    )
    for k in ["sharpe", "max_drawdown", "avg_turnover", "avg_cost", "total_return"]:
        assert np.isfinite(float(res.summary[k]))
