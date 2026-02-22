from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .evaluation import daily_ic, daily_rank_ic, max_drawdown, sharpe_ratio, summarize_series


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def _extract_metric_mean(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        return _safe_float(value.get("mean"))
    return _safe_float(value)


def _load_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _summarize_test_ic(preds: pd.DataFrame) -> Dict[str, Any]:
    if preds.empty:
        return {"rank_ic": {"n": 0, "mean": None, "std": None}, "ic": {"n": 0, "mean": None, "std": None}}
    test = preds[preds.get("split", "") == "test"].copy() if "split" in preds.columns else pd.DataFrame()
    if test.empty:
        return {"rank_ic": {"n": 0, "mean": None, "std": None}, "ic": {"n": 0, "mean": None, "std": None}}
    r = summarize_series(daily_rank_ic(test, score_col="y_pred", label_col="y_true"))
    p = summarize_series(daily_ic(test, score_col="y_pred", label_col="y_true"))
    return {"rank_ic": r, "ic": p}


def _summarize_backtest(daily: pd.DataFrame) -> Dict[str, Any]:
    if daily.empty:
        return {"days": 0, "sharpe": None, "max_drawdown": None, "total_return": None}
    sharpe = _safe_float(sharpe_ratio(pd.to_numeric(daily.get("net_return", pd.Series(dtype=float)), errors="coerce")))
    mdd = _safe_float(max_drawdown(pd.to_numeric(daily.get("equity", pd.Series(dtype=float)), errors="coerce")))
    total_return = None
    if "equity" in daily.columns and not daily["equity"].empty:
        total_return = _safe_float(pd.to_numeric(daily["equity"], errors="coerce").iloc[-1] - 1.0)
    return {
        "days": int(len(daily)),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "total_return": total_return,
    }


def build_auto_review(*, run_id: str, stage: str, run_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    model_test = (((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test") or {})
    bt_reported = (((metrics.get("backtest") or {}).get("summary")) or {})

    rank_ic_reported = _extract_metric_mean(model_test.get("rank_ic"))
    ic_reported = _extract_metric_mean(model_test.get("ic"))
    sharpe_reported = _safe_float(bt_reported.get("sharpe"))
    mdd_reported = _safe_float(bt_reported.get("max_drawdown"))
    total_return_reported = _safe_float(bt_reported.get("total_return"))

    preds = _load_parquet_if_exists(run_dir / "predictions.parquet")
    daily = _load_parquet_if_exists(run_dir / "backtest.parquet")
    ic_recomputed = _summarize_test_ic(preds)
    bt_recomputed = _summarize_backtest(daily)

    warnings: List[Dict[str, Any]] = []
    criticals: List[Dict[str, Any]] = []

    rank_ic_val = rank_ic_reported if rank_ic_reported is not None else _extract_metric_mean(ic_recomputed["rank_ic"])
    ic_val = ic_reported if ic_reported is not None else _extract_metric_mean(ic_recomputed["ic"])

    if rank_ic_val is not None and rank_ic_val <= 0.0:
        warnings.append(
            {
                "code": "rank_ic_non_positive",
                "message": "test rank_ic is non-positive; cross-sectional ranking signal may be weak or inverted",
                "value": rank_ic_val,
            }
        )
    if sharpe_reported is not None and sharpe_reported >= 8.0:
        warnings.append(
            {
                "code": "sharpe_extreme",
                "message": "reported sharpe is unusually high; verify annualization, costs, and leakage controls",
                "value": sharpe_reported,
            }
        )
    if sharpe_reported is not None and rank_ic_val is not None and sharpe_reported >= 4.0 and abs(rank_ic_val) < 0.01:
        warnings.append(
            {
                "code": "sharpe_rankic_mismatch",
                "message": "high sharpe with near-zero rank_ic indicates potential metric inconsistency",
                "sharpe": sharpe_reported,
                "rank_ic": rank_ic_val,
            }
        )
    if ic_val is not None and rank_ic_val is not None and abs(ic_val - rank_ic_val) >= 0.08:
        warnings.append(
            {
                "code": "ic_rankic_divergence",
                "message": "IC and rank_IC diverge materially; check outlier handling and label alignment",
                "ic": ic_val,
                "rank_ic": rank_ic_val,
            }
        )

    def _add_consistency_check(name: str, reported: Optional[float], recomputed: Optional[float], tol: float) -> None:
        if reported is None or recomputed is None:
            return
        if abs(reported - recomputed) > tol:
            criticals.append(
                {
                    "code": f"{name}_mismatch",
                    "message": f"{name} reported/recomputed mismatch exceeds tolerance",
                    "reported": reported,
                    "recomputed": recomputed,
                    "abs_diff": abs(reported - recomputed),
                    "tolerance": tol,
                }
            )

    _add_consistency_check("sharpe", sharpe_reported, _safe_float(bt_recomputed.get("sharpe")), 0.2)
    _add_consistency_check("max_drawdown", mdd_reported, _safe_float(bt_recomputed.get("max_drawdown")), 0.01)
    _add_consistency_check("total_return", total_return_reported, _safe_float(bt_recomputed.get("total_return")), 0.02)

    status = "pass"
    if criticals:
        status = "fail"
    elif warnings:
        status = "warn"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "stage": str(stage),
        "status": status,
        "warnings": warnings,
        "criticals": criticals,
        "metrics_reported": {
            "rank_ic_test_mean": rank_ic_reported,
            "ic_test_mean": ic_reported,
            "sharpe": sharpe_reported,
            "max_drawdown": mdd_reported,
            "total_return": total_return_reported,
        },
        "metrics_recomputed": {
            "test_ic": ic_recomputed,
            "backtest": bt_recomputed,
        },
    }


def write_auto_review(*, run_dir: Path, payload: Dict[str, Any]) -> Path:
    out = run_dir / "auto_review.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return out

