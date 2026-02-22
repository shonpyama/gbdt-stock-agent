from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


logger = logging.getLogger(__name__)

_NEWS_POSITIVE_TOKENS = {
    "beat",
    "beats",
    "bullish",
    "buy",
    "growth",
    "gain",
    "gains",
    "improve",
    "improves",
    "improved",
    "optimistic",
    "outperform",
    "strong",
    "upgrade",
    "upside",
}
_NEWS_NEGATIVE_TOKENS = {
    "bearish",
    "cut",
    "cuts",
    "decline",
    "declines",
    "downgrade",
    "drop",
    "drops",
    "fall",
    "falls",
    "miss",
    "misses",
    "risk",
    "risks",
    "warning",
    "weak",
}
_NEWS_TOKEN_RE = re.compile(r"[a-z']+")
_NEWS_TICKER_SPLIT_RE = re.compile(r"[,;\s|]+")
_NEWS_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _columns_hash(cols: Sequence[str]) -> str:
    return _sha1(",".join(sorted(cols)))[:12]


@dataclass(frozen=True)
class FeatureBuildResult:
    feature_store_id: str
    features: pd.DataFrame
    spec: Dict[str, Any]


def _pick_price_col(prices: pd.DataFrame, adjusted_flag: bool) -> str:
    if adjusted_flag and "adj_close" in prices.columns and prices["adj_close"].notna().any():
        return "adj_close"
    return "close"


def build_price_features(
    prices: pd.DataFrame,
    *,
    adjusted_flag: bool,
    lookbacks: Sequence[int],
) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    px_col = _pick_price_col(df, adjusted_flag)
    df["px"] = pd.to_numeric(df[px_col], errors="coerce")

    # Returns computed at date t use px_t. Shift by +1 to make them available for decision_date t+1.
    for lb in lookbacks:
        df[f"ret_{lb}d"] = df.groupby("symbol")["px"].pct_change(lb).shift(1)

    # Market proxy + rolling correlation to avoid overly local alpha.
    mkt = df.groupby("date", as_index=False)["px"].mean().rename(columns={"px": "mkt_px"})
    df = df.merge(mkt, on="date", how="left")
    df["mkt_ret_1d"] = df["mkt_px"].pct_change(1).shift(1)
    df["ret_1d_raw"] = df.groupby("symbol")["px"].pct_change(1)
    df["corr_mkt_20d"] = (
        df.groupby("symbol")
        .apply(
            lambda g: g["ret_1d_raw"].rolling(20, min_periods=20).corr(g["mkt_ret_1d"]).shift(1)
        )
        .reset_index(level=0, drop=True)
    )

    # Volatility (std of log returns over 20 days), shifted by +1.
    df["logret"] = np.log(df.groupby("symbol")["px"].pct_change(1) + 1.0)
    df["vol_20d"] = df.groupby("symbol")["logret"].rolling(20, min_periods=20).std().reset_index(level=0, drop=True).shift(1)

    # Trend / range / liquidity proxies
    if "high" in df.columns and "low" in df.columns:
        rng = (pd.to_numeric(df["high"], errors="coerce") - pd.to_numeric(df["low"], errors="coerce")) / df["px"].replace(0, np.nan)
        df["range_20d"] = rng.groupby(df["symbol"]).rolling(20, min_periods=20).mean().reset_index(level=0, drop=True).shift(1)
    else:
        df["range_20d"] = np.nan

    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").replace(0, np.nan)
        df["adv20_dollar"] = (vol * df["px"]).groupby(df["symbol"]).rolling(20, min_periods=20).mean().reset_index(level=0, drop=True).shift(1)
        df["volume_z_20d"] = (
            vol.groupby(df["symbol"]).rolling(20, min_periods=20).apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-12), raw=False)
        ).reset_index(level=0, drop=True).shift(1)
    else:
        df["adv20_dollar"] = np.nan
        df["volume_z_20d"] = np.nan

    if "ret_20d" in df.columns and "ret_60d" in df.columns:
        df["mom_20_60"] = df["ret_20d"] - df["ret_60d"]

    feature_cols = [c for c in df.columns if c.startswith(("ret_", "vol_", "range_", "adv", "volume_", "mom_", "corr_"))]
    out = df[["date", "symbol"] + feature_cols].rename(columns={"date": "decision_date"})

    # feature_available_date is previous trading row date (per-symbol), robust to holidays.
    out = out.sort_values(["symbol", "decision_date"]).reset_index(drop=True)
    out["feature_available_date"] = out.groupby("symbol")["decision_date"].shift(1)
    out = out[out["feature_available_date"].notna()].reset_index(drop=True)
    return out


def _prepare_earnings_features(
    earnings: pd.DataFrame,
    *,
    event_safe_shift_days: int,
) -> pd.DataFrame:
    if earnings.empty:
        return pd.DataFrame(columns=["decision_date", "symbol", "earnings_event", "eps_surprise"])

    df = earnings.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # Normalize date fields.
    for cand in ["date", "earningsDate", "epsDate", "reportDate"]:
        if cand in df.columns:
            df["event_date"] = pd.to_datetime(df[cand], errors="coerce").dt.date
            break
    else:
        df["event_date"] = pd.NaT

    df = df.dropna(subset=["event_date"])
    if df.empty:
        return pd.DataFrame(columns=["decision_date", "symbol", "earnings_event", "eps_surprise"])

    eff = (pd.to_datetime(df["event_date"]) + BDay(int(event_safe_shift_days))).dt.date
    df["decision_date"] = eff
    df["earnings_event"] = 1.0

    eps_act = None
    eps_est = None
    for k in ["eps", "epsActual", "epsReported", "epsactual"]:
        if k in df.columns:
            eps_act = pd.to_numeric(df[k], errors="coerce")
            break
    for k in ["epsEstimated", "epsEstimate", "epsestimated"]:
        if k in df.columns:
            eps_est = pd.to_numeric(df[k], errors="coerce")
            break
    if eps_act is not None and eps_est is not None:
        df["eps_surprise"] = eps_act - eps_est
    else:
        df["eps_surprise"] = np.nan

    df = df[["decision_date", "symbol", "earnings_event", "eps_surprise"]]
    df = df.groupby(["decision_date", "symbol"], as_index=False).agg({"earnings_event": "max", "eps_surprise": "mean"})
    return df


def _simple_lexicon_sentiment(text: str) -> float:
    toks = _NEWS_TOKEN_RE.findall(str(text).lower())
    if not toks:
        return float("nan")
    pos = sum(1 for t in toks if t in _NEWS_POSITIVE_TOKENS)
    neg = sum(1 for t in toks if t in _NEWS_NEGATIVE_TOKENS)
    return float((pos - neg) / max(1, len(toks)))


def _extract_news_symbol(value: Any) -> str:
    raw = str(value).upper().strip() if value is not None else ""
    if not raw:
        return "GENERAL"
    token = _NEWS_TICKER_SPLIT_RE.split(raw)[0].strip()
    if ":" in token:
        token = token.split(":")[-1].strip()
    token = token.replace("/", "").replace("\\", "").strip()
    if not token:
        return "GENERAL"
    return token if _NEWS_TICKER_RE.match(token) else "GENERAL"


def _prepare_news_features(
    news: pd.DataFrame,
    *,
    event_safe_shift_days: int,
) -> pd.DataFrame:
    cols = [
        "decision_date",
        "symbol",
        "news_count_1d",
        "news_sent_mean_1d",
        "news_source_nuniq_1d",
        "mkt_news_count_1d",
        "mkt_news_sent_mean_1d",
    ]
    if news.empty:
        return pd.DataFrame(columns=cols)

    df = news.copy()
    if "symbol" not in df.columns:
        if "tickers" in df.columns:
            df["symbol"] = df["tickers"].map(_extract_news_symbol)
        elif "ticker" in df.columns:
            df["symbol"] = df["ticker"].map(_extract_news_symbol)
        else:
            df["symbol"] = "GENERAL"
    elif "tickers" in df.columns:
        missing_symbol = df["symbol"].isna() | (df["symbol"].astype(str).str.strip() == "")
        if bool(missing_symbol.any()):
            df.loc[missing_symbol, "symbol"] = df.loc[missing_symbol, "tickers"].map(_extract_news_symbol)
    df["symbol"] = (
        df["symbol"]
        .fillna("GENERAL")
        .astype(str)
        .str.upper()
        .str.split(",")
        .str[0]
        .str.split(":")
        .str[-1]
        .str.strip()
        .replace("", "GENERAL")
    )

    event_ts = None
    for cand in ["publishedDate", "published_date", "date"]:
        if cand in df.columns:
            event_ts = pd.to_datetime(df[cand], errors="coerce", utc=True)
            if event_ts.notna().any():
                break
    if event_ts is None:
        return pd.DataFrame(columns=cols)

    df["event_date"] = event_ts.dt.tz_localize(None).dt.date
    df = df.dropna(subset=["event_date"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    sent = None
    for cand in ["sentiment", "sentimentScore", "sentiment_score"]:
        if cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce")
            if s.notna().any():
                sent = s.clip(-1.0, 1.0)
                break
    if sent is None:
        title = df["title"].astype(str) if "title" in df.columns else pd.Series("", index=df.index)
        body = df["text"].astype(str) if "text" in df.columns else pd.Series("", index=df.index)
        if "content" in df.columns:
            body = body.fillna("") + " " + df["content"].astype(str).fillna("")
        sent = (title.fillna("") + " " + body.fillna("")).map(_simple_lexicon_sentiment)
    df["news_sentiment"] = pd.to_numeric(sent, errors="coerce")

    if "site" in df.columns:
        df["news_source"] = df["site"].fillna("unknown").astype(str)
    elif "publisher" in df.columns:
        df["news_source"] = df["publisher"].fillna("unknown").astype(str)
    elif "source" in df.columns:
        df["news_source"] = df["source"].fillna("unknown").astype(str)
    else:
        df["news_source"] = "unknown"

    df["decision_date"] = (pd.to_datetime(df["event_date"]) + BDay(int(event_safe_shift_days))).dt.date

    per_symbol = (
        df.groupby(["decision_date", "symbol"], as_index=False)
        .agg(
            news_count_1d=("symbol", "size"),
            news_sent_mean_1d=("news_sentiment", "mean"),
            news_source_nuniq_1d=("news_source", "nunique"),
        )
        .reset_index(drop=True)
    )
    per_market = (
        df.groupby(["decision_date"], as_index=False)
        .agg(
            mkt_news_count_1d=("symbol", "size"),
            mkt_news_sent_mean_1d=("news_sentiment", "mean"),
        )
        .reset_index(drop=True)
    )
    out = per_symbol.merge(per_market, on="decision_date", how="left")
    out = out[out["symbol"] != "GENERAL"].reset_index(drop=True)
    return out[cols]


def build_feature_store(
    *,
    dataset_id: str,
    prices: pd.DataFrame,
    universe_membership: pd.DataFrame,
    earnings: Optional[pd.DataFrame],
    news: Optional[pd.DataFrame],
    adjusted_flag: bool,
    lookbacks: Sequence[int],
    event_safe_shift_days: int,
    out_dir: Path,
) -> FeatureBuildResult:
    price_feats = build_price_features(prices, adjusted_flag=adjusted_flag, lookbacks=lookbacks)
    if universe_membership.empty:
        raise RuntimeError("universe_membership is empty")

    mem = universe_membership.copy()
    mem["date"] = pd.to_datetime(mem["date"]).dt.date
    mem["symbol"] = mem["symbol"].astype(str).str.upper()
    mem = mem[mem["is_member"] == True][["date", "symbol"]].rename(columns={"date": "decision_date"})

    feats = price_feats.merge(mem, on=["decision_date", "symbol"], how="inner")

    if earnings is not None and not earnings.empty:
        e_feats = _prepare_earnings_features(earnings, event_safe_shift_days=event_safe_shift_days)
        feats = feats.merge(e_feats, on=["decision_date", "symbol"], how="left")
    else:
        feats["earnings_event"] = 0.0
        feats["eps_surprise"] = 0.0

    # No-event days should be 0 (not NaN), so training doesn't drop all rows.
    if "earnings_event" in feats.columns:
        feats["earnings_event"] = pd.to_numeric(feats["earnings_event"], errors="coerce").fillna(0.0)
    if "eps_surprise" in feats.columns:
        feats["eps_surprise"] = pd.to_numeric(feats["eps_surprise"], errors="coerce").fillna(0.0)

    if news is not None:
        n_feats = _prepare_news_features(news, event_safe_shift_days=event_safe_shift_days)
        if not n_feats.empty:
            feats = feats.merge(n_feats, on=["decision_date", "symbol"], how="left")

        base_cols = [
            "news_count_1d",
            "news_sent_mean_1d",
            "news_source_nuniq_1d",
            "mkt_news_count_1d",
            "mkt_news_sent_mean_1d",
        ]
        for c in base_cols:
            if c not in feats.columns:
                feats[c] = 0.0
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)

        feats = feats.sort_values(["symbol", "decision_date"]).reset_index(drop=True)
        feats["news_count_5d"] = (
            feats.groupby("symbol")["news_count_1d"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        feats["news_count_20d"] = (
            feats.groupby("symbol")["news_count_1d"].rolling(20, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        feats["news_sent_mean_5d"] = (
            feats.groupby("symbol")["news_sent_mean_1d"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        feats["news_sent_std_20d"] = (
            feats.groupby("symbol")["news_sent_mean_1d"].rolling(20, min_periods=2).std().reset_index(level=0, drop=True)
        )
        feats["mkt_news_count_20d"] = (
            feats.groupby("symbol")["mkt_news_count_1d"].rolling(20, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        feats["mkt_news_sent_mean_5d"] = (
            feats.groupby("symbol")["mkt_news_sent_mean_1d"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        feats["news_attention_20d"] = feats["news_count_20d"] / (feats["mkt_news_count_20d"] + 1e-12)

        for c in [
            "news_count_5d",
            "news_count_20d",
            "news_sent_mean_5d",
            "news_sent_std_20d",
            "mkt_news_count_20d",
            "mkt_news_sent_mean_5d",
            "news_attention_20d",
        ]:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)

    # Build spec + IDs
    feature_cols = [c for c in feats.columns if c not in {"decision_date", "symbol", "feature_available_date"}]
    spec = {
        "dataset_id": dataset_id,
        "lookbacks": list(lookbacks),
        "event_safe_shift_days": int(event_safe_shift_days),
        "include_news_features": bool(news is not None),
        "adjusted_flag": bool(adjusted_flag),
        "feature_cols": sorted(feature_cols),
    }
    columns_h = _columns_hash(feature_cols)
    feature_store_id = _sha1(json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + columns_h)[:12]

    feats["feature_version"] = columns_h
    feats = feats.sort_values(["decision_date", "symbol"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / feature_store_id).mkdir(parents=True, exist_ok=True)
    feats_path = out_dir / feature_store_id / "features.parquet"
    feats.to_parquet(feats_path, index=False)
    (out_dir / feature_store_id / "feature_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=True))

    return FeatureBuildResult(feature_store_id=feature_store_id, features=feats, spec=spec)
