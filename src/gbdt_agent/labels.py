from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelBuildResult:
    labels: pd.DataFrame
    summary: Dict[str, Any]


def build_future_return_labels(
    prices: pd.DataFrame,
    *,
    adjusted_flag: bool,
    horizon_trading_days: int,
) -> LabelBuildResult:
    if horizon_trading_days < 1:
        raise ValueError("horizon_trading_days must be >= 1")

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    px_col = "adj_close" if adjusted_flag and "adj_close" in df.columns and df["adj_close"].notna().any() else "close"
    df["px"] = pd.to_numeric(df[px_col], errors="coerce")

    df["label_date"] = df.groupby("symbol")["date"].shift(-horizon_trading_days)
    df["px_fwd"] = df.groupby("symbol")["px"].shift(-horizon_trading_days)
    target_col = f"future_return_{int(horizon_trading_days)}d"
    df[target_col] = (df["px_fwd"] / df["px"]) - 1.0

    labels = df.rename(columns={"date": "decision_date"})[
        ["decision_date", "label_date", "symbol", target_col]
    ].copy()
    labels["horizon"] = int(horizon_trading_days)

    labels = labels.dropna(subset=["label_date", target_col]).reset_index(drop=True)
    labels["label_date"] = pd.to_datetime(labels["label_date"]).dt.date
    labels["decision_date"] = pd.to_datetime(labels["decision_date"]).dt.date

    summary = {
        "horizon_trading_days": int(horizon_trading_days),
        "target_col": target_col,
        "rows": int(len(labels)),
        "nan_ratio": float(labels[target_col].isna().mean()),
    }
    return LabelBuildResult(labels=labels, summary=summary)
