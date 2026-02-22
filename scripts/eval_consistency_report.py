#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _daily_ic(preds: pd.DataFrame, *, rank: bool, winsor_q: float | None = None) -> Dict[str, Any]:
    vals = []
    for _, g in preds.groupby("decision_date"):
        y = pd.to_numeric(g["y_true"], errors="coerce")
        p = pd.to_numeric(g["y_pred"], errors="coerce")
        m = y.notna() & p.notna()
        if int(m.sum()) < 3:
            continue
        yy = y[m].copy()
        pp = p[m].copy()
        if winsor_q is not None and 0.0 < winsor_q < 0.5:
            lo = yy.quantile(winsor_q)
            hi = yy.quantile(1.0 - winsor_q)
            yy = yy.clip(lower=lo, upper=hi)
        method = "spearman" if rank else "pearson"
        ic = float(pd.Series(pp).corr(pd.Series(yy), method=method))
        if np.isfinite(ic):
            vals.append(ic)
    if not vals:
        return {"n": 0, "mean": None, "std": None}
    arr = np.asarray(vals, dtype=float)
    return {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}


def _sharpe_daily(returns: pd.Series, annualization: int = 252) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd <= 0.0:
        return float("nan")
    return float((r.mean() / sd) * np.sqrt(float(annualization)))


def _max_drawdown_from_equity(equity: pd.Series) -> float:
    x = pd.to_numeric(equity, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(x)
    dd = (x / peak) - 1.0
    return float(dd.min())


def main() -> int:
    parser = argparse.ArgumentParser(description="Create consistency diagnostics for one run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    run_dir = PROJECT_DIR / "artifacts" / "runs" / str(args.run_id)
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "predictions.parquet"
    daily_path = run_dir / "backtest.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(str(metrics_path))
    if not preds_path.exists():
        raise FileNotFoundError(str(preds_path))
    if not daily_path.exists():
        raise FileNotFoundError(str(daily_path))

    metrics = json.loads(metrics_path.read_text())
    preds = pd.read_parquet(preds_path)
    daily = pd.read_parquet(daily_path)
    test = preds[preds["split"] == "test"].copy()

    bt = ((metrics.get("backtest") or {}).get("summary") or {})
    sharpe_reported = _safe_float(bt.get("sharpe"))
    mdd_reported = _safe_float(bt.get("max_drawdown"))
    total_return_reported = _safe_float(bt.get("total_return"))
    sharpe_recomputed = _sharpe_daily(daily["net_return"], annualization=252)
    mdd_recomputed = _max_drawdown_from_equity(daily["equity"])
    total_return_recomputed = _safe_float(daily["equity"].iloc[-1] - 1.0) if not daily.empty else float("nan")

    out = {
        "run_id": str(args.run_id),
        "status": str(metrics.get("status")),
        "test_rows": int(len(test)),
        "test_dates": int(test["decision_date"].nunique()) if not test.empty else 0,
        "ic": {
            "pearson_test_reported": ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test", {}).get("ic"),
            "spearman_test_reported": ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test", {}).get("rank_ic"),
            "pearson_test_recomputed": _daily_ic(test, rank=False, winsor_q=None),
            "spearman_test_recomputed": _daily_ic(test, rank=True, winsor_q=None),
            "pearson_test_winsor_1pct": _daily_ic(test, rank=False, winsor_q=0.01),
            "spearman_test_winsor_1pct": _daily_ic(test, rank=True, winsor_q=0.01),
        },
        "backtest_consistency": {
            "reported": {
                "sharpe": sharpe_reported,
                "max_drawdown": mdd_reported,
                "total_return": total_return_reported,
                "avg_turnover": _safe_float(bt.get("avg_turnover")),
                "avg_cost": _safe_float(bt.get("avg_cost")),
                "days": int(bt.get("days") or 0),
            },
            "recomputed": {
                "sharpe": sharpe_recomputed,
                "max_drawdown": mdd_recomputed,
                "total_return": total_return_recomputed,
                "avg_turnover": _safe_float(daily.get("turnover", pd.Series(dtype=float)).mean()) if not daily.empty else float("nan"),
                "avg_cost": _safe_float(daily.get("cost_total", pd.Series(dtype=float)).mean()) if not daily.empty else float("nan"),
                "days": int(len(daily)),
            },
        },
    }

    out_path = Path(args.out_json) if str(args.out_json).strip() else (PROJECT_DIR / "reports" / f"eval_consistency_{args.run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

