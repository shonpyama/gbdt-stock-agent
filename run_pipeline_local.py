import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# 1. robust setup
try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False
except OSError:
    print("⚠️ LightGBM found but OS dependency (libomp) missing. Will use Sklearn.")
    HAS_LGB = False

if not HAS_LGB:
    from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def env_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    vals: list[float] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        try:
            vals.append(float(t))
        except ValueError:
            return default
    return vals if vals else default


# 2. Config
DEFAULT_API_KEY = "REPLACE_WITH_YOUR_KEY"
SYMBOLS = ["AAPL", "NVDA"]
BASE_DIR = os.path.join(os.getcwd(), "fmp_local_db")
OUT_REPORTS_DIR = os.path.join(os.getcwd(), "outputs", "reports")
OUT_CSV_DIR = os.path.join(os.getcwd(), "outputs", "csv")
FORWARD_DAYS = env_int("PIPE_FORWARD_DAYS", 20)
DEFAULT_SHARES = env_int("PIPE_DEFAULT_SHARES", 100)
TAX_MODE = os.environ.get("PIPE_TAX_MODE", "in").strip().lower() or "in"
EQUITY_SLIPPAGE_BPS = env_float("PIPE_EQUITY_SLIPPAGE_BPS", 5.0)
FX_SLIPPAGE_BPS = env_float("PIPE_FX_SLIPPAGE_BPS", 2.0)
RULE_THRESHOLDS = env_float_list("PIPE_RULE_THRESHOLDS", [0.50, 0.55, 0.60])
EMBARGO_DAYS = env_int("PIPE_EMBARGO_DAYS", 5)
MIN_RULE_TRADES = env_int("PIPE_MIN_RULE_TRADES", 300)
USE_GPU_IF_AVAILABLE = env_bool("PIPE_USE_GPU", True)
WALK_FORWARD_FOLDS = env_int("PIPE_WF_FOLDS", 3)
REQUIRE_LIVE_DATA = True
BASE_FEATURE_COLS = [
    "open",
    "volume",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "ma_10_dev",
    "ma_20_dev",
    "ma_50_dev",
    "range_pct",
    "vol_20d",
    "vol_ratio",
    "rsi_14",
    "atr_ratio",
]
MODEL_FEATURE_COLS = BASE_FEATURE_COLS + ["symbol_code"]


# 3. Helpers
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


[ensure_dir(os.path.join(BASE_DIR, d)) for d in ["meta", "raw/prices_eod", "reports"]]
[ensure_dir(OUT_REPORTS_DIR), ensure_dir(OUT_CSV_DIR)]


class FMP:
    def __init__(self, k: str):
        self.k = k
        self.base_url = "https://financialmodelingprep.com/stable"

    def get_price(self, s: str) -> pd.DataFrame:
        # Mock if key missing
        if self.k == DEFAULT_API_KEY:
            dates = pd.date_range("2023-01-01", periods=100)
            return pd.DataFrame(
                {
                    "date": dates,
                    "open": np.random.rand(100) * 100,
                    "high": np.random.rand(100) * 100,
                    "low": np.random.rand(100) * 100,
                    "close": np.random.rand(100) * 100,
                    "volume": np.random.randint(1000, 10000, 100),
                }
            )

        try:
            r = requests.get(
                f"{self.base_url}/historical-price-eod/full",
                params={"symbol": s, "apikey": self.k},
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or not payload:
                return pd.DataFrame()
            df = pd.DataFrame(payload)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.dropna(subset=["date", "open", "close", "volume"])
        except requests.RequestException:
            return pd.DataFrame()


def load_api_key() -> str:
    key = os.environ.get("FMP_API_KEY", "").strip()
    if key and "=" not in key:
        return key

    if key.upper().startswith("FMP_API_KEY="):
        parsed = key.split("=", 1)[1].strip().strip("'").strip('"')
        if parsed:
            return parsed

    key_file = Path("/content/.env_fmp")
    if key_file.exists():
        for raw in key_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("FMP_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("'").strip('"')
                if value:
                    return value
            elif "=" not in line:
                return line
    return DEFAULT_API_KEY


def get_symbols() -> list[str]:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return [s.strip().upper() for s in sys.argv[1].split(",") if s.strip()]
    return SYMBOLS


def safe_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    try:
        if y_true.nunique() < 2:
            return 0.0
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return 0.0


def calc_moomoo_fee_usd(notional_usd: float, tax_mode: str = "in") -> float:
    if notional_usd <= 0:
        return 0.0
    rate = 0.00132 if tax_mode == "in" else 0.0012
    cap = 22.0 if tax_mode == "in" else 20.0
    fee = notional_usd * rate
    return round(max(0.01, min(cap, fee)), 2)


def calc_trade_cost_bps(close_price: float) -> float:
    if close_price <= 0:
        return 0.0
    entry_notional = DEFAULT_SHARES * close_price
    exit_notional = entry_notional
    total_notional = entry_notional + exit_notional
    entry_fee = calc_moomoo_fee_usd(entry_notional, TAX_MODE)
    exit_fee = calc_moomoo_fee_usd(exit_notional, TAX_MODE)
    total_fee = entry_fee + exit_fee
    equity_slippage = total_notional * EQUITY_SLIPPAGE_BPS / 10000.0
    fx_slippage = total_notional * FX_SLIPPAGE_BPS / 10000.0
    total_cost = total_fee + equity_slippage + fx_slippage
    return float(total_cost / entry_notional * 10000.0)


def calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for p in [5, 10, 20, 60]:
        out[f"ret_{p}d"] = out["close"].pct_change(p)

    for p in [10, 20, 50]:
        sma = out["close"].rolling(p).mean()
        out[f"ma_{p}_dev"] = (out["close"] - sma) / (sma + 1e-10)

    out["range_pct"] = (out["high"] - out["low"]) / (out["close"] + 1e-10)
    out["vol_20d"] = out["close"].pct_change().rolling(20).std() * np.sqrt(252)
    out["vol_ratio"] = out["volume"] / (out["volume"].rolling(20).mean() + 1)
    out["rsi_14"] = ta.rsi(out["close"], length=14)
    out["atr_14"] = calc_atr(out["high"], out["low"], out["close"], 14)
    out["atr_ratio"] = out["atr_14"] / (out["close"] + 1e-10)
    return out


def make_model():
    if HAS_LGB:
        params = {"verbosity": -1, "random_state": 42}
        if USE_GPU_IF_AVAILABLE:
            params.update(
                {
                    "device_type": "gpu",
                    "max_bin": 63,
                }
            )
        return lgb.LGBMClassifier(**params)
    return RandomForestClassifier(n_estimators=10, random_state=42)


def fit_with_fallback(model, X_tr, y_tr):
    try:
        model.fit(X_tr, y_tr)
        return model, "gpu" if HAS_LGB and getattr(model, "device_type", "") == "gpu" else "cpu"
    except Exception as e:
        if HAS_LGB and isinstance(model, lgb.LGBMClassifier):
            cpu_model = lgb.LGBMClassifier(verbosity=-1, random_state=42, device_type="cpu")
            cpu_model.fit(X_tr, y_tr)
            print(f"GPU train fallback to CPU: {type(e).__name__}")
            return cpu_model, "cpu_fallback"
        raise


def apply_direction(proba: np.ndarray, direction: str) -> np.ndarray:
    if direction == "inverted":
        return 1.0 - proba
    return proba


def summarize_trade_returns(gross_rets: np.ndarray, net_rets: np.ndarray) -> dict:
    n_trades = int(len(net_rets))
    if n_trades == 0:
        return {
            "n_trades": 0,
            "total_orders": 0,
            "mean_gross_pct": 0.0,
            "mean_net_pct": 0.0,
            "sharpe_net": 0.0,
            "win_rate_pct": 0.0,
            "eligible": False,
        }
    mean_gross = float(np.mean(gross_rets)) * 100.0
    mean_net = float(np.mean(net_rets)) * 100.0
    std_net = float(np.std(net_rets))
    annual_factor = np.sqrt(252.0 / max(1.0, float(FORWARD_DAYS)))
    sharpe = float((np.mean(net_rets) / std_net) * annual_factor) if std_net > 1e-12 else 0.0
    win_rate = float(np.mean(net_rets > 0)) * 100.0
    return {
        "n_trades": n_trades,
        "total_orders": n_trades * 2,
        "mean_gross_pct": round(mean_gross, 4),
        "mean_net_pct": round(mean_net, 4),
        "sharpe_net": round(sharpe, 4),
        "win_rate_pct": round(win_rate, 2),
        "eligible": bool(n_trades >= MIN_RULE_TRADES),
    }


def select_best_rule(rule_stats: dict) -> tuple[str | None, float]:
    best_rule = None
    best_sharpe = -float("inf")
    for name, stat in rule_stats.items():
        if not bool(stat.get("eligible", True)):
            continue
        sharpe = float(stat.get("sharpe_net", 0.0))
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_rule = name

    if best_rule is None:
        return None, 0.50
    try:
        threshold = float(best_rule.split("_")[-1]) / 100.0
    except Exception:
        threshold = 0.50
    return best_rule, threshold


def run_walk_forward(full: pd.DataFrame, feature_cols: list[str], n_folds: int = 3):
    X_all = full[feature_cols]
    y_all = full["Target"]
    dates = np.sort(full["date"].unique())
    n_dates = len(dates)
    test_days = max(30, n_dates // (n_folds + 2))
    fold_metrics = []
    rule_names = [f"long_prob_{int(t * 100):02d}" for t in RULE_THRESHOLDS]
    directions = ["normal", "inverted"]
    rule_gross = {d: {name: [] for name in rule_names} for d in directions}
    rule_net = {d: {name: [] for name in rule_names} for d in directions}

    for fold in range(n_folds):
        test_start_idx = n_dates - (n_folds - fold) * test_days
        test_end_idx = min(test_start_idx + test_days, n_dates)
        if test_start_idx < 0 or test_end_idx <= test_start_idx:
            continue

        test_start_date = dates[test_start_idx]
        test_end_date = dates[test_end_idx - 1]
        embargo_date = test_start_date - np.timedelta64(EMBARGO_DAYS, "D")
        train_mask = full["date"] < embargo_date
        test_mask = (full["date"] >= test_start_date) & (full["date"] <= test_end_date)
        train_idx = full.index[train_mask].tolist()
        test_idx = full.index[test_mask].tolist()
        if len(train_idx) < 200 or len(test_idx) < 50:
            continue

        X_tr, y_tr = X_all.loc[train_idx], y_all.loc[train_idx]
        X_te, y_te = X_all.loc[test_idx], y_all.loc[test_idx]
        model = make_model()
        model, _ = fit_with_fallback(model, X_tr, y_tr)
        proba_raw = model.predict_proba(X_te)[:, 1]
        fold_df = full.loc[test_idx].copy()
        fold_result = {
            "fold": len(fold_metrics),
            "n_test": int(len(y_te)),
            "n_train": int(len(y_tr)),
            "test_start_date": str(pd.Timestamp(test_start_date).date()),
            "test_end_date": str(pd.Timestamp(test_end_date).date()),
        }
        for direction in directions:
            proba = apply_direction(proba_raw, direction)
            pred = (proba >= 0.5).astype(int)
            acc = float(np.mean(pred == y_te.values))
            auc = safe_auc(y_te, proba)
            fold_result[f"accuracy_{direction}"] = round(acc, 6)
            fold_result[f"auc_{direction}"] = round(auc, 6)
            for t in RULE_THRESHOLDS:
                rule = f"long_prob_{int(t * 100):02d}"
                sig = proba >= t
                rule_gross[direction][rule].extend(
                    fold_df.loc[sig, "fwd_ret_gross"].tolist()
                )
                rule_net[direction][rule].extend(fold_df.loc[sig, "fwd_ret_net"].tolist())
        fold_metrics.append(fold_result)

    if not fold_metrics:
        return [], 0.0, 0.0, {}, "normal", {}

    avg_auc_by_direction = {
        d: float(np.mean([f[f"auc_{d}"] for f in fold_metrics])) for d in directions
    }
    selected_direction = (
        "inverted"
        if avg_auc_by_direction["inverted"] > avg_auc_by_direction["normal"]
        else "normal"
    )
    avg_acc = float(
        np.mean([f[f"accuracy_{selected_direction}"] for f in fold_metrics])
    )
    avg_auc = float(np.mean([f[f"auc_{selected_direction}"] for f in fold_metrics]))
    results = [
        {
            "fold": int(f["fold"]),
            "accuracy": float(f[f"accuracy_{selected_direction}"]),
            "auc": float(f[f"auc_{selected_direction}"]),
            "n_test": int(f["n_test"]),
            "n_train": int(f["n_train"]),
            "test_start_date": f["test_start_date"],
            "test_end_date": f["test_end_date"],
        }
        for f in fold_metrics
    ]
    rule_stats = {
        rule: summarize_trade_returns(
            np.array(rule_gross[selected_direction][rule], dtype=float),
            np.array(rule_net[selected_direction][rule], dtype=float),
        )
        for rule in rule_gross[selected_direction]
    }
    diagnostics = {
        "avg_auc_normal": round(avg_auc_by_direction["normal"], 6),
        "avg_auc_inverted": round(avg_auc_by_direction["inverted"], 6),
    }
    return results, avg_acc, avg_auc, rule_stats, selected_direction, diagnostics


def run():
    print("Running Local Test...")
    api_key = load_api_key()
    if REQUIRE_LIVE_DATA and api_key == DEFAULT_API_KEY:
        raise RuntimeError(
            "FMP_API_KEY is required for live mode. Set env var or /content/.env_fmp."
        )
    symbols = get_symbols()
    client = FMP(api_key)
    print(f"Symbols: {symbols}")
    print("Data mode:", "mock" if api_key == DEFAULT_API_KEY else "live")

    # ETL
    dataset = []
    live_feature_frames = []
    for s in symbols:
        df = client.get_price(s)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
            df["symbol"] = s
            # Feat Eng
            if "close" in df.columns:
                df = add_features(df)
                df["fwd_ret_gross"] = np.log(df["close"].shift(-FORWARD_DAYS) / df["close"])
                df["cost_bps"] = df["close"].apply(calc_trade_cost_bps)
                df["fwd_ret_net"] = df["fwd_ret_gross"] - df["cost_bps"] / 10000.0
                df["Target"] = (df["fwd_ret_net"] > 0).astype(int)

                # live ranking pool: keep all rows with available real-time features
                feature_df = df.dropna(subset=BASE_FEATURE_COLS).copy()
                if not feature_df.empty:
                    keep_cols = list(
                        dict.fromkeys(
                            ["date", "symbol", "open", "high", "low", "close", "volume", "cost_bps"]
                            + BASE_FEATURE_COLS
                        )
                    )
                    live_feature_frames.append(
                        feature_df[keep_cols]
                    )

                model_df = df.dropna(
                    subset=BASE_FEATURE_COLS + ["fwd_ret_gross", "fwd_ret_net", "Target"]
                ).copy()
                if not model_df.empty:
                    dataset.append(model_df)

    # ML
    if dataset:
        full = pd.concat(dataset)
        full = full.sort_values(["date", "symbol"]).reset_index(drop=True)
        symbol_map = {s: i for i, s in enumerate(sorted(full["symbol"].unique()))}
        full["symbol_code"] = full["symbol"].map(symbol_map).astype(float)
        X = full[MODEL_FEATURE_COLS]
        y = full["Target"]
        unique_dates = np.sort(full["date"].unique())
        split_idx = int(len(unique_dates) * 0.8)
        if split_idx <= 0 or split_idx >= len(unique_dates):
            split_idx = max(1, len(unique_dates) - 1)
        train_end_date = unique_dates[split_idx - 1]
        test_start_date = unique_dates[split_idx]
        holdout_train_mask = full["date"] <= train_end_date
        holdout_test_mask = full["date"] >= test_start_date

        X_tr, y_tr = X.loc[holdout_train_mask], y.loc[holdout_train_mask]
        X_te, y_te = X.loc[holdout_test_mask], y.loc[holdout_test_mask]

        if HAS_LGB:
            print("Training LightGBM...")
        else:
            print("Training RandomForest (Fallback)...")
        clf = make_model()
        clf, holdout_device = fit_with_fallback(clf, X_tr, y_tr)
        proba_te_raw = clf.predict_proba(X_te)[:, 1]
        (
            wf_folds,
            wf_avg_acc,
            wf_avg_auc,
            wf_rule_stats,
            direction_mode,
            wf_diag,
        ) = run_walk_forward(full, MODEL_FEATURE_COLS, n_folds=WALK_FORWARD_FOLDS)
        selected_rule, decision_threshold = select_best_rule(wf_rule_stats)
        proba_te = apply_direction(proba_te_raw, direction_mode)
        pred_te = (proba_te >= 0.5).astype(int)
        score = float(np.mean(pred_te == y_te.values))
        auc = safe_auc(y_te, proba_te)
        print(f"Test Score: {score:.4f}")
        print(f"Test AUC: {auc:.4f}")
        print(f"Train device (holdout): {holdout_device}")
        print(f"Direction mode: {direction_mode}")
        print(f"Walk-Forward Folds: {len(wf_folds)}")
        print(f"WF Avg Accuracy: {wf_avg_acc:.4f}")
        print(f"WF Avg AUC: {wf_avg_auc:.4f}")

        # Plot
        plt.figure()
        plt.plot(y_te.values, label="True")
        plt.plot(clf.predict(X_te), label="Pred", alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(BASE_DIR, "reports", "local_test.png"))
        print("Analysis generated.")

        clf_live = make_model()
        clf_live, live_device = fit_with_fallback(clf_live, X, y)
        print(f"Train device (live): {live_device}")

        if live_feature_frames:
            live_pool = pd.concat(live_feature_frames, ignore_index=True)
            date_coverage = live_pool.groupby("date")["symbol"].nunique()
            common_dates = date_coverage[date_coverage == len(symbols)]
            if len(common_dates) > 0:
                latest_date = common_dates.index.max()
                latest_rows = live_pool[live_pool["date"] == latest_date].copy()
            else:
                best_cov = int(date_coverage.max()) if len(date_coverage) else 0
                latest_date = date_coverage[date_coverage == best_cov].index.max()
                latest_rows = live_pool[live_pool["date"] == latest_date].copy()
                print(
                    f"Warning: no full common date for all symbols; "
                    f"using date={pd.Timestamp(latest_date).date()} coverage={best_cov}/{len(symbols)}"
                )
            latest_rows = (
                latest_rows.sort_values(["date", "symbol"])
                .groupby("symbol", as_index=False)
                .tail(1)
                .copy()
            )
        else:
            latest_rows = full.sort_values("date").groupby("symbol", as_index=False).tail(1).copy()
        latest_rows["symbol_code"] = latest_rows["symbol"].map(symbol_map).astype(float)
        latest_X = latest_rows[MODEL_FEATURE_COLS]
        latest_rows["pred_proba_raw"] = clf_live.predict_proba(latest_X)[:, 1]
        latest_rows["pred_proba"] = apply_direction(
            latest_rows["pred_proba_raw"].values, direction_mode
        )
        latest_rows["ev_score"] = latest_rows["pred_proba"] - 0.5
        latest_rows["fwd_ret_20d_net_pct"] = latest_rows["ev_score"] * 100.0
        latest_rows["selected_threshold"] = decision_threshold
        latest_rows["action"] = np.where(
            latest_rows["pred_proba"] >= decision_threshold, "LONG", "FLAT"
        )
        latest_rows = latest_rows.sort_values("ev_score", ascending=False).reset_index(
            drop=True
        )
        latest_rows["rank"] = latest_rows.index + 1

        fi = []
        if HAS_LGB and hasattr(clf_live, "feature_importances_"):
            names = MODEL_FEATURE_COLS
            vals = clf_live.feature_importances_.astype(float)
            denom = float(vals.sum()) if float(vals.sum()) > 0 else 1.0
            for i, (n, v) in enumerate(
                sorted(
                    zip(names, vals / denom),
                    key=lambda x: x[1],
                    reverse=True,
                ),
                start=1,
            ):
                fi.append({"rank": i, "feature": n, "importance": round(float(v), 6)})

        report = {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_symbols": int(len(symbols)),
                "tax_mode": TAX_MODE,
                "fee_rate": "0.132%",
                "fee_cap": "N/A",
                "slippage_equity_bps": EQUITY_SLIPPAGE_BPS,
                "slippage_fx_bps": FX_SLIPPAGE_BPS,
                "debug_mode": False,
                "score_definition": {
                    "ev_score": "pred_proba - 0.5 (proxy expected edge, no future return used)",
                    "pred_proba": "classifier probability for 20-day net return > 0",
                    "ranking_basis": "ev_score desc",
                    "direction_mode": direction_mode,
                    "selected_rule": selected_rule,
                    "decision_threshold": decision_threshold,
                },
                "run_config": {
                    "symbols": symbols,
                    "forward_days": FORWARD_DAYS,
                    "walk_forward_folds": WALK_FORWARD_FOLDS,
                    "rule_thresholds": RULE_THRESHOLDS,
                    "embargo_days": EMBARGO_DAYS,
                    "gpu_requested": USE_GPU_IF_AVAILABLE,
                    "train_device_holdout": holdout_device,
                    "train_device_live": live_device,
                },
                "diagnostics": wf_diag,
            },
            "walk_forward": {
                "folds": [
                    {
                        "fold": int(f["fold"]),
                        "accuracy": float(f["accuracy"]),
                        "auc": float(f["auc"]),
                        "n_test": int(f["n_test"]),
                    }
                    for f in wf_folds
                ],
                "avg_accuracy": round(float(wf_avg_acc), 6),
                "avg_auc": round(float(wf_avg_auc), 6),
            },
            "exit_rules_comparison": wf_rule_stats,
            "feature_importance": fi,
            "top_candidates": [
                {
                    "rank": int(r["rank"]),
                    "symbol": str(r["symbol"]),
                    "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                    "pred_proba": round(float(r["pred_proba"]), 6),
                    "pred_proba_raw": round(float(r["pred_proba_raw"]), 6),
                    "model_signal_pct": round(float(r["pred_proba"]) * 100.0, 2),
                    "ev_score": round(float(r["ev_score"]), 6),
                    "ev_score_basis": "pred_proba_minus_0_5",
                    "decision_threshold": round(float(r["selected_threshold"]), 4),
                    "action": str(r["action"]),
                    "fwd_ret_20d_net_pct": round(float(r["fwd_ret_20d_net_pct"]), 4),
                    "cost_bps": round(float(r["cost_bps"]), 1),
                }
                for _, r in latest_rows.head(10).iterrows()
            ],
        }

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(OUT_REPORTS_DIR, f"claude_summary_{ts}.json")
        csv_path = os.path.join(OUT_CSV_DIR, f"candidates_{ts}.csv")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        latest_rows.to_csv(csv_path, index=False)
        print(f"Saved report: {report_path}")
        print(f"Saved candidates: {csv_path}")
    else:
        print("No data was loaded. Check API key/symbols.")

    print("Success.")


if __name__ == "__main__":
    run()
