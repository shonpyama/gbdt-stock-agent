from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    summary: Dict[str, Any]
    excluded_symbols: List[str]


def validate_raw_prices(
    prices: pd.DataFrame,
    *,
    adjusted_flag: bool,
    thresholds: Dict[str, Any],
    splits: Optional[pd.DataFrame] = None,
    dividends: Optional[pd.DataFrame] = None,
) -> ValidationResult:
    """
    Validation Gate (Stage 1):
    - missing ratio
    - duplicates
    - OHLC anomalies
    - volume anomalies
    - symbol min observations
    - adjusted consistency
    """
    if prices.empty:
        return ValidationResult(False, {"error": "prices_empty"}, [])

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    summary: Dict[str, Any] = {}

    # Missing ratio (numeric columns)
    num_cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    missing_by_col = {c: float(df[c].isna().mean()) for c in num_cols}
    summary["missing_by_col"] = missing_by_col
    summary["missing_ratio_overall"] = float(df[num_cols].isna().mean().mean()) if num_cols else 0.0

    # Duplicates
    dup_count = int(df.duplicated(subset=["date", "symbol"]).sum())
    dup_ratio = float(dup_count / max(1, len(df)))
    summary["duplicates_count"] = dup_count
    summary["duplicates_ratio"] = dup_ratio

    # OHLC anomalies
    anomalies = 0
    if {"open", "high", "low", "close"}.issubset(df.columns):
        ohlc = df[["open", "high", "low", "close"]]
        neg = (ohlc < 0).any(axis=1)
        hl = df["high"] < df["low"]
        close_out = (df["close"] < df["low"]) | (df["close"] > df["high"])
        anomalies = int((neg | hl | close_out).sum())
    summary["ohlc_anomaly_rows"] = anomalies
    summary["ohlc_anomaly_ratio"] = float(anomalies / len(df)) if len(df) else 0.0

    # Volume anomalies
    vol0_ratio = float((df["volume"] == 0).mean()) if "volume" in df.columns else None
    summary["volume_zero_ratio"] = vol0_ratio

    # Zero volume streaks (per symbol).
    zero_streak_len = int(thresholds.get("volume_zero_streak_len", 5))
    if "volume" in df.columns and zero_streak_len > 1:
        zdf = df.sort_values(["symbol", "date"])[["symbol", "date", "volume"]].copy()
        zdf["is_zero"] = (pd.to_numeric(zdf["volume"], errors="coerce").fillna(0.0) <= 0).astype(int)
        bad_rows = 0
        for _, g in zdf.groupby("symbol"):
            s = g["is_zero"].to_numpy(dtype=int)
            if len(s) < zero_streak_len:
                continue
            k = zero_streak_len
            rolling = np.convolve(s, np.ones(k, dtype=int), mode="valid")
            bad_rows += int((rolling >= k).sum())
        summary["zero_volume_streak_len"] = zero_streak_len
        summary["zero_volume_streak_violations"] = bad_rows
        summary["zero_volume_streak_violation_ratio"] = float(bad_rows / max(1, len(df)))
    else:
        summary["zero_volume_streak_len"] = zero_streak_len
        summary["zero_volume_streak_violations"] = 0
        summary["zero_volume_streak_violation_ratio"] = 0.0

    # Extreme spikes (abs return > threshold)
    jump_thr = float(thresholds.get("price_jump_threshold", 0.2))
    if "close" in df.columns:
        df = df.sort_values(["symbol", "date"])
        df["ret1"] = df.groupby("symbol")["close"].pct_change(1)
        jump_ratio = float((df["ret1"].abs() > jump_thr).mean(skipna=True))
        summary["price_jump_ratio"] = jump_ratio
    else:
        summary["price_jump_ratio"] = None

    # Min observations per symbol
    min_obs = int(thresholds.get("min_obs_per_symbol", 252))
    obs = df.groupby("symbol")["date"].count()
    excluded = sorted(obs[obs < min_obs].index.tolist())
    summary["symbols_total"] = int(obs.shape[0])
    summary["symbols_excluded_min_obs"] = int(len(excluded))
    summary["min_obs_threshold"] = min_obs

    # Adjusted consistency
    if adjusted_flag:
        if "adj_close" not in df.columns:
            summary["adj_close_missing"] = True
            passed_adj = False
        else:
            adj_missing_ratio = float(df["adj_close"].isna().mean())
            summary["adj_close_missing_ratio"] = adj_missing_ratio
            passed_adj = adj_missing_ratio <= float(thresholds.get("adjclose_missing_ratio_max", 0.02))

            # Corporate-action consistency (best-effort):
            # adjusted price should reduce discontinuity around split/dividend dates.
            split_viol_ratio = 0.0
            div_viol_ratio = 0.0
            if {"close", "adj_close", "symbol", "date"}.issubset(df.columns):
                px = df.sort_values(["symbol", "date"])[["symbol", "date", "close", "adj_close"]].copy()
                px["close_ret1"] = px.groupby("symbol")["close"].pct_change(1).abs()
                px["adj_ret1"] = px.groupby("symbol")["adj_close"].pct_change(1).abs()

                if splits is not None and not splits.empty and {"symbol", "date"}.issubset(splits.columns):
                    sp = splits.copy()
                    sp["symbol"] = sp["symbol"].astype(str).str.upper()
                    sp["date"] = pd.to_datetime(sp["date"]).dt.date
                    sp = sp.drop_duplicates(subset=["symbol", "date"])
                    m = px.merge(sp[["symbol", "date"]], on=["symbol", "date"], how="inner")
                    if not m.empty:
                        # Expect adjusted move not to be much larger than raw move near split day.
                        viol = (m["adj_ret1"] > (m["close_ret1"] * float(thresholds.get("split_adj_multiplier_max", 1.2)) + 1e-8)).sum()
                        split_viol_ratio = float(viol / len(m))

                if dividends is not None and not dividends.empty and {"symbol", "date"}.issubset(dividends.columns):
                    dv = dividends.copy()
                    dv["symbol"] = dv["symbol"].astype(str).str.upper()
                    dv["date"] = pd.to_datetime(dv["date"]).dt.date
                    dv = dv.drop_duplicates(subset=["symbol", "date"])
                    m = px.merge(dv[["symbol", "date"]], on=["symbol", "date"], how="inner")
                    if not m.empty:
                        viol = (m["adj_ret1"] > m["close_ret1"] + float(thresholds.get("div_adj_extra_tolerance", 0.03))).sum()
                        div_viol_ratio = float(viol / len(m))

            summary["split_adjustment_violation_ratio"] = split_viol_ratio
            summary["dividend_adjustment_violation_ratio"] = div_viol_ratio
            if split_viol_ratio > float(thresholds.get("split_adjustment_violation_ratio_max", 0.3)):
                passed_adj = False
            if div_viol_ratio > float(thresholds.get("dividend_adjustment_violation_ratio_max", 0.3)):
                passed_adj = False
    else:
        passed_adj = True

    # Pass/fail thresholds
    missing_max = float(thresholds.get("missing_ratio_max", 0.05))
    anomaly_max = float(thresholds.get("anomaly_row_ratio_max", 0.02))
    duplicates_max = float(thresholds.get("duplicates_ratio_max", 0.0))
    zv_streak_max = float(thresholds.get("volume_zero_streak_ratio_max", 0.02))
    min_symbols_after = int(thresholds.get("min_symbols_after_filter", 20))

    passed = True
    reasons: List[str] = []
    if summary["missing_ratio_overall"] > missing_max:
        passed = False
        reasons.append(f"missing_ratio_overall={summary['missing_ratio_overall']:.4f}>{missing_max}")
    if summary["ohlc_anomaly_ratio"] > anomaly_max:
        passed = False
        reasons.append(f"ohlc_anomaly_ratio={summary['ohlc_anomaly_ratio']:.4f}>{anomaly_max}")
    if summary["duplicates_ratio"] > duplicates_max:
        passed = False
        reasons.append(f"duplicates_ratio={summary['duplicates_ratio']:.4f}>{duplicates_max}")
    if summary["zero_volume_streak_violation_ratio"] > zv_streak_max:
        passed = False
        reasons.append(
            f"zero_volume_streak_violation_ratio={summary['zero_volume_streak_violation_ratio']:.4f}>{zv_streak_max}"
        )
    symbols_after_filter = max(0, int(summary["symbols_total"]) - int(summary["symbols_excluded_min_obs"]))
    summary["symbols_after_filter"] = symbols_after_filter
    summary["min_symbols_after_filter"] = min_symbols_after
    if symbols_after_filter < min_symbols_after:
        passed = False
        reasons.append(f"symbols_after_filter={symbols_after_filter}<{min_symbols_after}")
    if not passed_adj:
        passed = False
        reasons.append("adjusted_consistency_failed")

    summary["passed"] = passed
    summary["fail_reasons"] = reasons

    return ValidationResult(passed=passed, summary=summary, excluded_symbols=excluded)
