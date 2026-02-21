from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None  # type: ignore


def daily_rank_ic(df: pd.DataFrame, *, score_col: str = "y_pred", label_col: str = "y_true") -> List[float]:
    ics: List[float] = []
    for _, g in df.groupby("decision_date"):
        s = pd.to_numeric(g[score_col], errors="coerce")
        y = pd.to_numeric(g[label_col], errors="coerce")
        m = s.notna() & y.notna()
        if int(m.sum()) < 3:
            continue
        if spearmanr is not None:
            ic = float(spearmanr(s[m].to_numpy(), y[m].to_numpy()).correlation)
        else:
            ic = float(pd.Series(s[m]).corr(pd.Series(y[m]), method="spearman"))
        if np.isfinite(ic):
            ics.append(ic)
    return ics


def daily_ic(df: pd.DataFrame, *, score_col: str = "y_pred", label_col: str = "y_true") -> List[float]:
    ics: List[float] = []
    for _, g in df.groupby("decision_date"):
        s = pd.to_numeric(g[score_col], errors="coerce")
        y = pd.to_numeric(g[label_col], errors="coerce")
        m = s.notna() & y.notna()
        if int(m.sum()) < 3:
            continue
        ic = float(pd.Series(s[m]).corr(pd.Series(y[m]), method="pearson"))
        if np.isfinite(ic):
            ics.append(ic)
    return ics


def summarize_series(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    arr = np.asarray(xs, dtype=float)
    return {"n": int(len(xs)), "mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(xs) > 1 else 0.0}


def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    r = returns.dropna().to_numpy(dtype=float)
    if r.size < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(annualization))


def max_drawdown(equity: pd.Series) -> float:
    x = equity.to_numpy(dtype=float)
    if x.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(x)
    dd = (x / peak) - 1.0
    return float(dd.min())


def evaluate_predictions(preds: pd.DataFrame) -> Dict[str, Any]:
    """
    preds columns: decision_date, symbol, y_true, y_pred, model_name, split
    """
    out: Dict[str, Any] = {}
    for (model_name, split), g in preds.groupby(["model_name", "split"]):
        rank_ics = daily_rank_ic(g, score_col="y_pred", label_col="y_true")
        ics = daily_ic(g, score_col="y_pred", label_col="y_true")
        out.setdefault(model_name, {})[split] = {
            "rank_ic_daily": rank_ics,
            "ic_daily": ics,
            "rank_ic": summarize_series(rank_ics),
            "ic": summarize_series(ics),
            **summarize_series(rank_ics),
        }
    return out
