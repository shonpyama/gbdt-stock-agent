from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluation import max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame
    summary: Dict[str, Any]


def _cap_and_normalize(weights: pd.Series, cap: float) -> pd.Series:
    if weights.empty:
        return weights
    if cap <= 0:
        cap = 1.0
    w = weights.clip(upper=cap)
    s = float(w.sum())
    if s <= 0:
        return w
    return w / s


def run_backtest(
    preds: pd.DataFrame,
    *,
    topn: int,
    long_short: bool,
    max_names: int,
    single_name_cap: float,
    rebalance: str,
    commission_bps: float,
    slippage_base_bps: float,
    slippage_k_adv: float,
    slippage_k_vol: float,
    notional: float = 1_000_000.0,
) -> BacktestResult:
    """
    preds must include:
      decision_date, symbol, y_true, y_pred
    Optional columns for cost model:
      adv20_dollar, vol_20d
    """
    df = preds.copy()
    df["decision_date"] = pd.to_datetime(df["decision_date"]).dt.date
    df = df.sort_values(["decision_date", "symbol"]).reset_index(drop=True)

    prev_w: pd.Series = pd.Series(dtype=float)
    last_rebalance_token: Optional[tuple[int, int]] = None
    daily_rows = []
    pos_rows = []

    def _safe_mean(v: pd.Series) -> float:
        if not isinstance(v, pd.Series):
            return 0.0
        x = pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            return 0.0
        return float(x.mean())

    for d, g in df.groupby("decision_date"):
        g = g.copy()
        g["y_pred"] = pd.to_numeric(g["y_pred"], errors="coerce")
        g["y_true"] = pd.to_numeric(g["y_true"], errors="coerce")
        g = g.dropna(subset=["y_pred", "y_true"])
        if g.empty:
            continue

        rebalance_mode = str(rebalance or "daily").lower()
        if rebalance_mode not in {"daily", "weekly"}:
            raise ValueError(f"Unsupported rebalance mode: {rebalance_mode}")
        token = (pd.Timestamp(d).isocalendar().year, pd.Timestamp(d).isocalendar().week) if rebalance_mode == "weekly" else (0, 0)
        do_rebalance = (rebalance_mode == "daily") or (last_rebalance_token != token) or prev_w.empty

        if do_rebalance:
            n_long = min(int(topn), int(max_names), len(g))
            g = g.sort_values("y_pred", ascending=False)
            longs = g.head(n_long).copy()
            longs["weight"] = 1.0 / max(1, n_long)

            if long_short:
                n_short = min(int(topn), int(max_names) - n_long, len(g) - n_long)
                shorts = g.tail(n_short).copy()
                shorts["weight"] = -1.0 / max(1, n_short)
                pos = pd.concat([longs, shorts], ignore_index=True)
            else:
                pos = longs

            # Enforce single-name cap (absolute), then renormalize by gross exposure for long_short.
            cap = float(single_name_cap)
            if long_short:
                w = pos.set_index("symbol")["weight"]
                w = w.clip(lower=-cap, upper=cap)
                gross = float(w.abs().sum())
                if gross > 0:
                    w = w / gross
                pos = pos.drop(columns=["weight"]).merge(w.rename("weight").reset_index(), on="symbol", how="left")
            else:
                w = _cap_and_normalize(pos.set_index("symbol")["weight"], cap)
                pos = pos.drop(columns=["weight"]).merge(w.rename("weight").reset_index(), on="symbol", how="left")

            w_now = pos.set_index("symbol")["weight"]
            # Turnover based on union index.
            union_idx = prev_w.index.union(w_now.index)
            delta = (w_now.reindex(union_idx, fill_value=0.0) - prev_w.reindex(union_idx, fill_value=0.0)).abs().sum()
            turnover = float(delta / 2.0)
            last_rebalance_token = token
        else:
            w_now = prev_w.copy()
            turnover = 0.0
            pos = g[g["symbol"].isin(w_now.index)].copy()
            if pos.empty:
                continue
            pos = pos.drop_duplicates(subset=["symbol"], keep="last")
            pos = pos.merge(w_now.rename("weight").reset_index().rename(columns={"index": "symbol"}), on="symbol", how="inner")

        # Costs: commission + slippage (simple, feature-driven).
        commission = (commission_bps / 10000.0) * turnover

        pos_idx = pos.set_index("symbol", drop=False)
        adv = (
            pd.to_numeric(pos_idx["adv20_dollar"], errors="coerce").reindex(w_now.index)
            if "adv20_dollar" in pos_idx.columns
            else pd.Series([np.nan] * len(w_now), index=w_now.index)
        )
        vol = (
            pd.to_numeric(pos_idx["vol_20d"], errors="coerce").reindex(w_now.index)
            if "vol_20d" in pos_idx.columns
            else pd.Series([np.nan] * len(w_now), index=w_now.index)
        )

        # Approx trade dollars per symbol: |delta_w| * notional (use current delta vs prev).
        dw = (w_now - prev_w.reindex(w_now.index, fill_value=0.0)).abs().rename("abs_delta_w")
        trade_dollar = dw * float(notional)
        adv_safe = adv.fillna(1e12).replace(0, 1e12)
        vol_safe = vol.fillna(0.0)
        impact_adv = _safe_mean(trade_dollar / adv_safe)
        impact_vol = _safe_mean(vol_safe)
        slip_bps = float(slippage_base_bps) + float(slippage_k_adv) * impact_adv + float(slippage_k_vol) * impact_vol
        slippage = (slip_bps / 10000.0) * turnover

        total_cost = float(commission + slippage)

        gross = float((pos["weight"] * pos["y_true"]).sum())
        net = gross - total_cost

        daily_rows.append({
            "date": d,
            "gross_return": gross,
            "net_return": net,
            "turnover": turnover,
            "commission": commission,
            "slippage": slippage,
            "cost_total": total_cost,
            "n_names": int(len(pos)),
            "slip_bps_est": slip_bps,
        })

        pos_out = pos[["symbol", "weight", "y_pred", "y_true"]].copy()
        pos_out.insert(0, "date", d)
        pos_rows.append(pos_out)

        prev_w = w_now

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return BacktestResult(daily=daily, positions=pd.DataFrame(), summary={"error": "no_trading_days"})

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0

    summary = {
        "days": int(len(daily)),
        "sharpe": sharpe_ratio(daily["net_return"]),
        "max_drawdown": max_drawdown(daily["equity"]),
        "avg_turnover": float(daily["turnover"].mean()),
        "avg_cost": float(daily["cost_total"].mean()),
        "total_return": float(daily["equity"].iloc[-1] - 1.0),
    }

    positions = pd.concat(pos_rows, ignore_index=True) if pos_rows else pd.DataFrame()
    return BacktestResult(daily=daily, positions=positions, summary=summary)
