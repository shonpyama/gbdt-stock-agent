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
_NEWS_SYMBOL_PLACEHOLDERS = {"GENERAL", "MARKET", "ALL", "GLOBAL"}
_NUMERIC_VALUE_RE = re.compile(r"[-+]?\d*\.?\d+")
_NEWS_TOPIC_SPECS: Dict[str, Dict[str, Sequence[str]]] = {
    "earnings": {
        "tokens": ("earnings", "eps", "revenue", "beat", "miss"),
        "phrases": ("q1 results", "q2 results", "q3 results", "q4 results"),
    },
    "guidance": {
        "tokens": ("guidance", "outlook", "forecast", "projection", "guides"),
        "phrases": ("raises guidance", "cuts guidance", "lowers guidance", "full-year outlook"),
    },
    "mna": {
        "tokens": ("merger", "acquisition", "acquire", "acquires", "takeover", "buyout"),
        "phrases": ("m&a", "to acquire", "to buy", "strategic transaction"),
    },
    "litigation": {
        "tokens": ("lawsuit", "litigation", "court", "trial", "settlement", "sue", "sued"),
        "phrases": ("legal action", "class action", "court ruling"),
    },
    "regulation": {
        "tokens": ("regulation", "regulatory", "antitrust", "compliance", "ban", "sanction", "probe"),
        "phrases": ("regulatory approval", "regulatory filing", "government investigation"),
    },
    "product": {
        "tokens": ("product", "launch", "release", "rollout", "roadmap", "feature", "platform"),
        "phrases": ("new product", "product launch", "new model"),
    },
    "supply_chain": {
        "tokens": ("supplier", "suppliers", "inventory", "logistics", "shortage", "shipment", "capacity"),
        "phrases": ("supply chain", "component shortage", "production delay", "factory shutdown"),
    },
    "macro": {
        "tokens": ("inflation", "cpi", "ppi", "gdp", "rates", "fed", "fomc", "recession"),
        "phrases": ("interest rate", "economic growth", "monetary policy"),
    },
    "geopolitics": {
        "tokens": ("war", "conflict", "geopolitical", "tariff", "border", "election", "embargo"),
        "phrases": ("trade war", "geopolitical tension", "military action"),
    },
    "analyst": {
        "tokens": ("analyst", "upgrade", "downgrade", "outperform", "underperform", "overweight", "target"),
        "phrases": ("price target", "rating action", "broker note"),
    },
    "consensus": {
        "tokens": ("consensus", "estimate", "estimates", "expectation", "expects"),
        "phrases": ("street estimate", "analyst consensus"),
    },
    "employment": {
        "tokens": ("hiring", "hired", "layoff", "layoffs", "headcount", "workforce", "jobs", "unemployment"),
        "phrases": ("job cuts", "job growth", "labor market"),
    },
}
_NEWS_TOPIC_KEYS = tuple(_NEWS_TOPIC_SPECS.keys())
_GRADE_SCORE_RULES: List[Tuple[str, float]] = [
    ("strong buy", 2.0),
    ("conviction buy", 2.0),
    ("buy", 1.0),
    ("outperform", 1.0),
    ("overweight", 1.0),
    ("market outperform", 0.75),
    ("accumulate", 0.5),
    ("hold", 0.0),
    ("neutral", 0.0),
    ("market perform", 0.0),
    ("equal weight", 0.0),
    ("underperform", -1.0),
    ("underweight", -1.0),
    ("reduce", -1.0),
    ("sell", -2.0),
    ("strong sell", -2.0),
]
_GRADE_ACTION_BONUS: List[Tuple[str, float]] = [
    ("upgrade", 0.5),
    ("reiterate", 0.0),
    ("maintain", 0.0),
    ("downgrade", -0.5),
]


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


def _grade_text_score(grade: Any, action: Any = None) -> float:
    raw = str(grade).strip().lower()
    if not raw or raw in {"nan", "none", "null"}:
        return float("nan")
    score = None
    for token, val in _GRADE_SCORE_RULES:
        if token in raw:
            score = float(val)
            break
    if score is None:
        return float("nan")

    act = str(action).strip().lower()
    if act and act not in {"nan", "none", "null"}:
        for token, bonus in _GRADE_ACTION_BONUS:
            if token in act:
                score += float(bonus)
                break
    return float(score)


def _news_topic_hits(text: str) -> Dict[str, float]:
    raw = str(text).lower()
    tokens = set(_NEWS_TOKEN_RE.findall(raw))
    out: Dict[str, float] = {}
    for key, spec in _NEWS_TOPIC_SPECS.items():
        token_hit = any(str(t).lower() in tokens for t in spec.get("tokens", ()))
        phrase_hit = any(str(p).lower() in raw for p in spec.get("phrases", ()))
        out[key] = 1.0 if (token_hit or phrase_hit) else 0.0
    return out


def _extract_news_symbols(value: Any, *, drop_placeholders: bool = False) -> List[str]:
    raw = str(value).upper().strip() if value is not None else ""
    if not raw or raw in {"NAN", "NONE"}:
        return []
    tokens = _NEWS_TICKER_SPLIT_RE.split(raw.replace("/", " ").replace("\\", " "))
    out: List[str] = []
    seen = set()
    for tok in tokens:
        token = tok.strip()
        if not token:
            continue
        if ":" in token:
            token = token.split(":")[-1].strip()
        if not token or (drop_placeholders and token in _NEWS_SYMBOL_PLACEHOLDERS) or not _NEWS_TICKER_RE.match(token):
            continue
        if token in seen:
            continue
        out.append(token)
        seen.add(token)
    return out


def _extract_news_symbol(value: Any) -> str:
    symbols = _extract_news_symbols(value)
    if symbols:
        return symbols[0]
    return "GENERAL"


def _prepare_news_features(
    news: pd.DataFrame,
    *,
    event_safe_shift_days: int,
    general_symbol_uses_tickers: bool = False,
) -> pd.DataFrame:
    topic_symbol_cols = [f"news_topic_{k}_1d" for k in _NEWS_TOPIC_KEYS]
    topic_market_cols = [f"mkt_news_topic_{k}_1d" for k in _NEWS_TOPIC_KEYS]
    cols = [
        "decision_date",
        "symbol",
        "news_count_1d",
        "news_sent_mean_1d",
        "news_source_nuniq_1d",
        "mkt_news_count_1d",
        "mkt_news_sent_mean_1d",
    ] + topic_symbol_cols + topic_market_cols
    if news.empty:
        return pd.DataFrame(columns=cols)

    df = news.copy()

    event_ts = None
    for cand in ["publishedDate", "published_date", "date"]:
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], errors="coerce", utc=True)
            if event_ts is None:
                event_ts = ts
            else:
                # Keep the first available timestamp per row across known date columns.
                event_ts = event_ts.fillna(ts)
    if event_ts is None or not bool(event_ts.notna().any()):
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

    title_txt = df["title"].astype(str) if "title" in df.columns else pd.Series("", index=df.index)
    body_txt = pd.Series("", index=df.index, dtype=object)
    if "text" in df.columns:
        body_txt = body_txt.fillna("") + " " + df["text"].astype(str).fillna("")
    if "content" in df.columns:
        body_txt = body_txt.fillna("") + " " + df["content"].astype(str).fillna("")
    news_topic_payload = (title_txt.fillna("") + " " + body_txt.fillna("")).map(_news_topic_hits)
    for key in _NEWS_TOPIC_KEYS:
        src_key = key
        out_col = f"news_topic_{key}_1d"
        df[out_col] = news_topic_payload.map(lambda d, k=src_key: float(d.get(k, 0.0)))

    if "site" in df.columns:
        df["news_source"] = df["site"].fillna("unknown").astype(str)
    elif "publisher" in df.columns:
        df["news_source"] = df["publisher"].fillna("unknown").astype(str)
    elif "source" in df.columns:
        df["news_source"] = df["source"].fillna("unknown").astype(str)
    else:
        df["news_source"] = "unknown"

    df["decision_date"] = (pd.to_datetime(df["event_date"]) + BDay(int(event_safe_shift_days))).dt.date

    # Build symbol-level assignments with fallback order:
    # symbol -> tickers -> ticker -> GENERAL.
    symbol_lists = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
    if "symbol" in df.columns:
        symbol_lists = df["symbol"].map(
            lambda v: _extract_news_symbols(v, drop_placeholders=bool(general_symbol_uses_tickers))
        )
    if "tickers" in df.columns:
        tickers_lists = df["tickers"].map(lambda v: _extract_news_symbols(v, drop_placeholders=True))
        symbol_lists = pd.Series(
            [s if len(s) > 0 else t for s, t in zip(symbol_lists.tolist(), tickers_lists.tolist())],
            index=df.index,
            dtype=object,
        )
    if "ticker" in df.columns:
        ticker_lists = df["ticker"].map(lambda v: _extract_news_symbols(v, drop_placeholders=True))
        symbol_lists = pd.Series(
            [s if len(s) > 0 else t for s, t in zip(symbol_lists.tolist(), ticker_lists.tolist())],
            index=df.index,
            dtype=object,
        )
    df["symbol_list"] = symbol_lists.map(lambda xs: xs if xs else ["GENERAL"])

    market_agg: Dict[str, Tuple[str, str]] = {
        "mkt_news_count_1d": ("decision_date", "size"),
        "mkt_news_sent_mean_1d": ("news_sentiment", "mean"),
    }
    market_agg.update(
        {
            f"mkt_news_topic_{k}_1d": (f"news_topic_{k}_1d", "sum")
            for k in _NEWS_TOPIC_KEYS
        }
    )
    per_market = df.groupby(["decision_date"], as_index=False).agg(**market_agg).reset_index(drop=True)

    symbol_df = (
        df[["decision_date", "symbol_list", "news_sentiment", "news_source"] + topic_symbol_cols]
        .explode("symbol_list")
        .rename(columns={"symbol_list": "symbol"})
        .reset_index(drop=True)
    )
    symbol_df["symbol"] = symbol_df["symbol"].fillna("GENERAL").astype(str).str.upper().str.strip().replace("", "GENERAL")

    symbol_agg: Dict[str, Tuple[str, str]] = {
        "news_count_1d": ("symbol", "size"),
        "news_sent_mean_1d": ("news_sentiment", "mean"),
        "news_source_nuniq_1d": ("news_source", "nunique"),
    }
    symbol_agg.update({f"news_topic_{k}_1d": (f"news_topic_{k}_1d", "sum") for k in _NEWS_TOPIC_KEYS})
    per_symbol = symbol_df.groupby(["decision_date", "symbol"], as_index=False).agg(**symbol_agg).reset_index(drop=True)
    out = per_symbol.merge(per_market, on="decision_date", how="left")
    return out[cols]


def _parse_numeric_value(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        if isinstance(value, (int, float, np.number)):
            return float(value)
        s = str(value).strip()
        if not s:
            return float("nan")
        s = s.replace(",", "")
        if s.lower() in {"nan", "none", "null", "na", "n/a", "--"}:
            return float("nan")
        m = _NUMERIC_VALUE_RE.search(s)
        if not m:
            return float("nan")
        return float(m.group(0))
    except Exception:
        return float("nan")


def _coalesce_numeric_cols(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for c in candidates:
        if c not in frame.columns:
            continue
        vals = pd.to_numeric(frame[c], errors="coerce")
        out = out.where(out.notna(), vals)
        if bool(out.notna().all()):
            break
    return out


def _prepare_financials_features(
    financials: pd.DataFrame,
    *,
    event_safe_shift_days: int,
) -> pd.DataFrame:
    cols = [
        "decision_date",
        "symbol",
        "fin_gross_margin",
        "fin_operating_margin",
        "fin_net_margin",
        "fin_ebitda_margin",
        "fin_current_ratio",
        "fin_debt_to_assets",
        "fin_cash_to_debt",
        "fin_roe",
        "fin_roa",
        "fin_fcf_margin",
        "fin_revenue_yoy",
        "fin_eps_yoy",
        "fin_assets_qoq",
        "fin_debt_qoq",
    ]
    if financials.empty:
        return pd.DataFrame(columns=cols)
    if "symbol" not in financials.columns or "statement_type" not in financials.columns:
        return pd.DataFrame(columns=cols)

    df = financials.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    for cand in ("date", "fillingDate", "acceptedDate"):
        if cand in df.columns:
            df["event_date"] = pd.to_datetime(df[cand], errors="coerce").dt.date
            break
    else:
        return pd.DataFrame(columns=cols)
    df = df[df["event_date"].notna()].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=cols)

    stype = df["statement_type"].astype(str).str.lower()

    def _prepare_stmt(stmt_name: str, field_map: Dict[str, Sequence[str]]) -> pd.DataFrame:
        part = df[stype == stmt_name].copy()
        if part.empty:
            return pd.DataFrame(columns=["event_date", "symbol"] + list(field_map.keys()))
        out = part[["event_date", "symbol"]].copy()
        for out_col, cands in field_map.items():
            out[out_col] = _coalesce_numeric_cols(part, cands)
        out = out.groupby(["event_date", "symbol"], as_index=False).last()
        return out

    income = _prepare_stmt(
        "income",
        {
            "revenue": ["revenue", "totalRevenue"],
            "gross_profit": ["grossProfit"],
            "operating_income": ["operatingIncome", "operatingIncomeLoss"],
            "net_income": ["netIncome", "netIncomeApplicableToCommonShares"],
            "ebitda": ["ebitda", "EBITDA"],
            "eps": ["eps", "epsdiluted", "epsDiluted"],
        },
    )
    balance = _prepare_stmt(
        "balance",
        {
            "total_assets": ["totalAssets"],
            "total_current_assets": ["totalCurrentAssets"],
            "total_current_liabilities": ["totalCurrentLiabilities"],
            "total_debt": ["totalDebt", "shortLongTermDebtTotal", "longTermDebt"],
            "cash_and_equiv": ["cashAndCashEquivalents", "cashAndCashEquivalentsAtCarryingValue", "cashAndShortTermInvestments"],
            "total_equity": ["totalStockholdersEquity", "totalShareholderEquity", "totalEquity"],
        },
    )
    cashflow = _prepare_stmt(
        "cashflow",
        {
            "operating_cf": ["operatingCashFlow", "netCashProvidedByOperatingActivities"],
            "capex": ["capitalExpenditure", "capitalExpenditures"],
            "free_cf": ["freeCashFlow"],
        },
    )

    panel = income.merge(balance, on=["event_date", "symbol"], how="outer").merge(cashflow, on=["event_date", "symbol"], how="outer")
    if panel.empty:
        return pd.DataFrame(columns=cols)

    eps = 1e-12
    panel = panel.sort_values(["symbol", "event_date"]).reset_index(drop=True)

    def _series(name: str) -> pd.Series:
        if name in panel.columns:
            return pd.to_numeric(panel[name], errors="coerce")
        return pd.Series(np.nan, index=panel.index, dtype=float)

    if "free_cf" in panel.columns and "operating_cf" in panel.columns and "capex" in panel.columns:
        fcf_fallback = panel["operating_cf"] - panel["capex"].abs()
        panel["free_cf"] = panel["free_cf"].where(pd.notna(panel["free_cf"]), fcf_fallback)

    revenue = _series("revenue")
    gross_profit = _series("gross_profit")
    operating_income = _series("operating_income")
    net_income = _series("net_income")
    ebitda = _series("ebitda")
    total_current_assets = _series("total_current_assets")
    total_current_liabilities = _series("total_current_liabilities")
    total_debt = _series("total_debt")
    total_assets = _series("total_assets")
    cash_and_equiv = _series("cash_and_equiv")
    total_equity = _series("total_equity")
    free_cf = _series("free_cf")

    panel["fin_gross_margin"] = gross_profit / (revenue.abs() + eps)
    panel["fin_operating_margin"] = operating_income / (revenue.abs() + eps)
    panel["fin_net_margin"] = net_income / (revenue.abs() + eps)
    panel["fin_ebitda_margin"] = ebitda / (revenue.abs() + eps)
    panel["fin_current_ratio"] = total_current_assets / (total_current_liabilities.abs() + eps)
    panel["fin_debt_to_assets"] = total_debt / (total_assets.abs() + eps)
    panel["fin_cash_to_debt"] = cash_and_equiv / (total_debt.abs() + eps)
    panel["fin_roe"] = net_income / (total_equity.abs() + eps)
    panel["fin_roa"] = net_income / (total_assets.abs() + eps)
    panel["fin_fcf_margin"] = free_cf / (revenue.abs() + eps)
    panel["fin_revenue_yoy"] = panel.groupby("symbol")["revenue"].pct_change(4, fill_method=None)
    panel["fin_eps_yoy"] = panel.groupby("symbol")["eps"].pct_change(4, fill_method=None)
    panel["fin_assets_qoq"] = panel.groupby("symbol")["total_assets"].pct_change(1, fill_method=None)
    panel["fin_debt_qoq"] = panel.groupby("symbol")["total_debt"].pct_change(1, fill_method=None)

    panel["decision_date"] = (pd.to_datetime(panel["event_date"]) + BDay(int(event_safe_shift_days))).dt.date
    out_cols = [c for c in cols if c in panel.columns]
    out = panel[out_cols].copy()
    for c in out.columns:
        if c in {"decision_date", "symbol"}:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.groupby(["decision_date", "symbol"], as_index=False).last().sort_values(["decision_date", "symbol"]).reset_index(drop=True)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def _prepare_macro_treasury_features(
    macro_treasury: pd.DataFrame,
    *,
    event_safe_shift_days: int,
) -> pd.DataFrame:
    base = [
        "treasury_month1",
        "treasury_month2",
        "treasury_month3",
        "treasury_month6",
        "treasury_year1",
        "treasury_year2",
        "treasury_year3",
        "treasury_year5",
        "treasury_year7",
        "treasury_year10",
        "treasury_year20",
        "treasury_year30",
        "treasury_spread_10y_2y",
        "treasury_spread_10y_3m",
        "treasury_spread_30y_5y",
    ]
    cols = ["decision_date"] + base
    if macro_treasury.empty:
        return pd.DataFrame(columns=cols)

    df = macro_treasury.copy()
    if "date" not in df.columns:
        return pd.DataFrame(columns=cols)
    df["event_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["event_date"].notna()].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=cols)

    level_cols = [c for c in ["month1", "month2", "month3", "month6", "year1", "year2", "year3", "year5", "year7", "year10", "year20", "year30"] if c in df.columns]
    if not level_cols:
        return pd.DataFrame(columns=cols)
    for c in level_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df[["event_date"] + level_cols].copy()
    out["decision_date"] = (pd.to_datetime(out["event_date"]) + BDay(int(event_safe_shift_days))).dt.date
    out = out.drop(columns=["event_date"])
    out = out.groupby("decision_date", as_index=False).mean(numeric_only=True)
    out = out.rename(columns={c: f"treasury_{c}" for c in level_cols})

    if "treasury_year10" in out.columns and "treasury_year2" in out.columns:
        out["treasury_spread_10y_2y"] = out["treasury_year10"] - out["treasury_year2"]
    else:
        out["treasury_spread_10y_2y"] = np.nan
    if "treasury_year10" in out.columns and "treasury_month3" in out.columns:
        out["treasury_spread_10y_3m"] = out["treasury_year10"] - out["treasury_month3"]
    else:
        out["treasury_spread_10y_3m"] = np.nan
    if "treasury_year30" in out.columns and "treasury_year5" in out.columns:
        out["treasury_spread_30y_5y"] = out["treasury_year30"] - out["treasury_year5"]
    else:
        out["treasury_spread_30y_5y"] = np.nan

    for c in base:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out[cols].sort_values(["decision_date"]).reset_index(drop=True)


def _prepare_macro_calendar_features(
    macro_calendar: pd.DataFrame,
    *,
    event_safe_shift_days: int,
) -> pd.DataFrame:
    cols = [
        "decision_date",
        "mkt_macro_events_us_1d",
        "mkt_macro_high_impact_us_1d",
        "mkt_macro_surprise_mean_1d",
        "mkt_macro_surprise_abs_mean_1d",
    ]
    if macro_calendar.empty:
        return pd.DataFrame(columns=cols)

    df = macro_calendar.copy()
    if "date" not in df.columns:
        return pd.DataFrame(columns=cols)
    ts = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["event_date"] = ts.dt.tz_localize(None).dt.date
    df = df[df["event_date"].notna()].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=cols)

    if "country" in df.columns:
        c = df["country"].astype(str).str.upper().str.strip()
        df = df[c.isin({"US", "USA"})].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=cols)

    impact = df["impact"].astype(str).str.lower() if "impact" in df.columns else pd.Series([""] * len(df), index=df.index)
    df["impact_high"] = impact.str.contains("high", na=False).astype(float)
    actual = df["actual"].map(_parse_numeric_value) if "actual" in df.columns else pd.Series(np.nan, index=df.index)
    estimate = df["estimate"].map(_parse_numeric_value) if "estimate" in df.columns else pd.Series(np.nan, index=df.index)
    df["macro_surprise"] = pd.to_numeric(actual - estimate, errors="coerce")
    df["macro_surprise_abs"] = np.abs(df["macro_surprise"])
    df["decision_date"] = (pd.to_datetime(df["event_date"]) + BDay(int(event_safe_shift_days))).dt.date

    out = (
        df.groupby(["decision_date"], as_index=False)
        .agg(
            mkt_macro_events_us_1d=("decision_date", "size"),
            mkt_macro_high_impact_us_1d=("impact_high", "sum"),
            mkt_macro_surprise_mean_1d=("macro_surprise", "mean"),
            mkt_macro_surprise_abs_mean_1d=("macro_surprise_abs", "mean"),
        )
        .reset_index(drop=True)
    )
    return out[cols]


def _prepare_analyst_features(
    analyst: pd.DataFrame,
    *,
    event_safe_shift_days: int,
) -> pd.DataFrame:
    cols = [
        "decision_date",
        "symbol",
        "analyst_grade_score_1d",
        "analyst_grade_events_1d",
        "analyst_grade_score_20d",
        "analyst_grade_events_20d",
        "analyst_consensus_net_1d",
        "analyst_consensus_buy_ratio_1d",
        "analyst_consensus_sell_ratio_1d",
        "analyst_consensus_net_5d",
        "analyst_target_consensus_1d",
        "analyst_target_spread_1d",
        "analyst_target_analysts_1d",
        "analyst_target_spread_20d",
        "analyst_eps_estimate_1d",
        "analyst_revenue_estimate_1d",
        "analyst_eps_estimate_rev_20d",
        "analyst_rating_score_1d",
    ]
    if analyst.empty or "symbol" not in analyst.columns:
        return pd.DataFrame(columns=cols)

    df = analyst.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    src = df["source"].astype(str).str.lower().str.strip() if "source" in df.columns else pd.Series([""] * len(df), index=df.index)
    df["source"] = src

    event_ts = None
    for cand in ("date", "publishedDate", "published_date", "gradingDate", "updatedAt", "lastUpdated", "fiscalDateEnding"):
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], errors="coerce", utc=True)
            event_ts = ts if event_ts is None else event_ts.fillna(ts)
    if event_ts is None:
        return pd.DataFrame(columns=cols)
    snapshot_mask = pd.Series(False, index=df.index)
    if bool(event_ts.isna().any()):
        # Some premium endpoints are snapshot-style (no date field). Treat as-of fetch date to avoid backfilling into past.
        snapshot_sources = {"grades_consensus", "price_target_summary", "price_target_consensus"}
        snapshot_mask = event_ts.isna() & df["source"].isin(snapshot_sources)
        if bool(snapshot_mask.any()):
            event_ts = event_ts.copy()
            event_ts.loc[snapshot_mask] = pd.Timestamp.now(tz="UTC") - BDay(1)
    if not bool(event_ts.notna().any()):
        return pd.DataFrame(columns=cols)

    df["event_date"] = event_ts.dt.tz_localize(None).dt.date
    df["is_snapshot_asof"] = snapshot_mask.astype(bool)
    df = df[df["event_date"].notna()].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["decision_date"] = (pd.to_datetime(df["event_date"]) + BDay(int(event_safe_shift_days))).dt.date
    if bool(df["is_snapshot_asof"].any()):
        # Snapshot endpoints are queried at run-time and treated as available immediately.
        df.loc[df["is_snapshot_asof"], "decision_date"] = pd.to_datetime(
            df.loc[df["is_snapshot_asof"], "event_date"],
            errors="coerce",
        ).dt.date

    grade_df = df[df["source"] == "grades"].copy()
    grade_out = pd.DataFrame(columns=["decision_date", "symbol", "analyst_grade_score_1d", "analyst_grade_events_1d"])
    if not grade_df.empty:
        grade_col = pd.Series("", index=grade_df.index, dtype=object)
        action_col = pd.Series("", index=grade_df.index, dtype=object)
        for cand in ("newGrade", "grade", "ratingRecommendation", "recommendation", "rating", "toGrade"):
            if cand in grade_df.columns:
                grade_col = grade_col.where(grade_col.astype(str).str.strip() != "", grade_df[cand].astype(str))
        for cand in ("action", "gradingAction", "ratingChange"):
            if cand in grade_df.columns:
                action_col = action_col.where(action_col.astype(str).str.strip() != "", grade_df[cand].astype(str))
        grade_df["analyst_grade_score_1d"] = [
            _grade_text_score(g, a) for g, a in zip(grade_col.tolist(), action_col.tolist())
        ]
        grade_df["analyst_grade_events_1d"] = 1.0
        grade_out = (
            grade_df.groupby(["decision_date", "symbol"], as_index=False)
            .agg(
                analyst_grade_score_1d=("analyst_grade_score_1d", "mean"),
                analyst_grade_events_1d=("analyst_grade_events_1d", "sum"),
            )
            .reset_index(drop=True)
        )

    cons_df = df[df["source"].isin({"grades_consensus", "grades_historical"})].copy()
    cons_out = pd.DataFrame(
        columns=["decision_date", "symbol", "analyst_consensus_net_1d", "analyst_consensus_buy_ratio_1d", "analyst_consensus_sell_ratio_1d"]
    )
    if not cons_df.empty:
        strong_buy = _coalesce_numeric_cols(
            cons_df,
            ["strongBuy", "strong_buy", "strongBuyCount", "strongBuyRatings", "analystRatingsStrongBuy"],
        )
        buy = _coalesce_numeric_cols(cons_df, ["buy", "buyCount", "buyRatings", "analystRatingsBuy"])
        hold = _coalesce_numeric_cols(
            cons_df,
            ["hold", "holdCount", "holdRatings", "neutral", "neutralCount", "analystRatingsHold"],
        )
        sell = _coalesce_numeric_cols(cons_df, ["sell", "sellCount", "sellRatings", "analystRatingsSell"])
        strong_sell = _coalesce_numeric_cols(
            cons_df,
            ["strongSell", "strong_sell", "strongSellCount", "strongSellRatings", "analystRatingsStrongSell"],
        )
        total = strong_buy.fillna(0.0) + buy.fillna(0.0) + hold.fillna(0.0) + sell.fillna(0.0) + strong_sell.fillna(0.0)
        total_safe = total.where(total > 0.0, np.nan)
        cons_df["analyst_consensus_buy_ratio_1d"] = (strong_buy.fillna(0.0) + buy.fillna(0.0)) / total_safe
        cons_df["analyst_consensus_sell_ratio_1d"] = (sell.fillna(0.0) + strong_sell.fillna(0.0)) / total_safe
        cons_df["analyst_consensus_net_1d"] = (
            (strong_buy.fillna(0.0) + buy.fillna(0.0) - sell.fillna(0.0) - strong_sell.fillna(0.0)) / total_safe
        )
        cons_out = (
            cons_df.groupby(["decision_date", "symbol"], as_index=False)
            .agg(
                analyst_consensus_net_1d=("analyst_consensus_net_1d", "mean"),
                analyst_consensus_buy_ratio_1d=("analyst_consensus_buy_ratio_1d", "mean"),
                analyst_consensus_sell_ratio_1d=("analyst_consensus_sell_ratio_1d", "mean"),
            )
            .reset_index(drop=True)
        )

    target_df = df[df["source"].isin({"price_target_summary", "price_target_consensus"})].copy()
    target_out = pd.DataFrame(columns=["decision_date", "symbol", "analyst_target_consensus_1d", "analyst_target_spread_1d", "analyst_target_analysts_1d"])
    if not target_df.empty:
        consensus = _coalesce_numeric_cols(
            target_df,
            [
                "targetConsensus",
                "targetPrice",
                "priceTarget",
                "priceTargetConsensus",
                "consensusPriceTarget",
                "targetMean",
                "avgPriceTarget",
                "targetMedian",
                "lastMonthAvgPriceTarget",
                "lastQuarterAvgPriceTarget",
                "lastYearAvgPriceTarget",
                "allTimeAvgPriceTarget",
            ],
        )
        high = _coalesce_numeric_cols(target_df, ["targetHigh", "priceTargetHigh", "highPriceTarget", "high"])
        low = _coalesce_numeric_cols(target_df, ["targetLow", "priceTargetLow", "lowPriceTarget", "low"])
        analysts = _coalesce_numeric_cols(
            target_df,
            ["numberOfAnalysts", "numAnalysts", "analystCount", "analysts", "lastMonthCount", "lastQuarterCount", "lastYearCount", "allTimeCount"],
        )
        target_df["analyst_target_consensus_1d"] = consensus
        target_df["analyst_target_spread_1d"] = (high - low) / (consensus.abs() + 1e-12)
        target_df["analyst_target_analysts_1d"] = analysts
        target_out = (
            target_df.groupby(["decision_date", "symbol"], as_index=False)
            .agg(
                analyst_target_consensus_1d=("analyst_target_consensus_1d", "mean"),
                analyst_target_spread_1d=("analyst_target_spread_1d", "mean"),
                analyst_target_analysts_1d=("analyst_target_analysts_1d", "mean"),
            )
            .reset_index(drop=True)
        )

    est_df = df[df["source"] == "analyst_estimates"].copy()
    est_out = pd.DataFrame(columns=["decision_date", "symbol", "analyst_eps_estimate_1d", "analyst_revenue_estimate_1d"])
    if not est_df.empty:
        est_df["analyst_eps_estimate_1d"] = _coalesce_numeric_cols(
            est_df,
            ["estimatedEpsAvg", "estimatedEPSAvg", "estimatedEps", "epsEstimated", "epsEstimate", "consensusEPS", "epsAvg"],
        )
        est_df["analyst_revenue_estimate_1d"] = _coalesce_numeric_cols(
            est_df,
            ["estimatedRevenueAvg", "estimatedRevenue", "consensusRevenue", "revenueEstimate", "revenueAvg"],
        )
        est_out = (
            est_df.groupby(["decision_date", "symbol"], as_index=False)
            .agg(
                analyst_eps_estimate_1d=("analyst_eps_estimate_1d", "mean"),
                analyst_revenue_estimate_1d=("analyst_revenue_estimate_1d", "mean"),
            )
            .reset_index(drop=True)
        )

    rating_df = df[df["source"] == "ratings_historical"].copy()
    rating_out = pd.DataFrame(columns=["decision_date", "symbol", "analyst_rating_score_1d"])
    if not rating_df.empty:
        rating_df["analyst_rating_score_1d"] = _coalesce_numeric_cols(rating_df, ["rating", "ratingScore", "score", "overallScore"])
        score_cols = [c for c in rating_df.columns if str(c).endswith("Score")]
        if score_cols:
            score_frame = rating_df[score_cols].apply(pd.to_numeric, errors="coerce")
            score_mean = score_frame.mean(axis=1, skipna=True)
            rating_df["analyst_rating_score_1d"] = rating_df["analyst_rating_score_1d"].where(
                rating_df["analyst_rating_score_1d"].notna(),
                score_mean,
            )
        rating_out = (
            rating_df.groupby(["decision_date", "symbol"], as_index=False)
            .agg(analyst_rating_score_1d=("analyst_rating_score_1d", "mean"))
            .reset_index(drop=True)
        )

    merged: Optional[pd.DataFrame] = None
    for part in [grade_out, cons_out, target_out, est_out, rating_out]:
        if part is None or part.empty:
            continue
        merged = part if merged is None else merged.merge(part, on=["decision_date", "symbol"], how="outer")
    if merged is None:
        return pd.DataFrame(columns=cols)

    out = merged.sort_values(["symbol", "decision_date"]).reset_index(drop=True)
    for c in [k for k in ["analyst_grade_score_1d", "analyst_grade_events_1d"] if k in out.columns]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    for c in [
        "analyst_consensus_net_1d",
        "analyst_consensus_buy_ratio_1d",
        "analyst_consensus_sell_ratio_1d",
        "analyst_target_consensus_1d",
        "analyst_target_spread_1d",
        "analyst_target_analysts_1d",
        "analyst_eps_estimate_1d",
        "analyst_revenue_estimate_1d",
        "analyst_rating_score_1d",
    ]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out.groupby("symbol")[c].ffill().fillna(0.0)

    out["analyst_grade_score_20d"] = (
        out.groupby("symbol")["analyst_grade_score_1d"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["analyst_grade_events_20d"] = (
        out.groupby("symbol")["analyst_grade_events_1d"].rolling(20, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    out["analyst_consensus_net_5d"] = (
        out.groupby("symbol")["analyst_consensus_net_1d"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["analyst_target_spread_20d"] = (
        out.groupby("symbol")["analyst_target_spread_1d"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["analyst_eps_estimate_rev_20d"] = out.groupby("symbol")["analyst_eps_estimate_1d"].pct_change(20)

    for c in cols:
        if c in {"decision_date", "symbol"}:
            continue
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out[cols].sort_values(["decision_date", "symbol"]).reset_index(drop=True)


def build_feature_store(
    *,
    dataset_id: str,
    prices: pd.DataFrame,
    universe_membership: pd.DataFrame,
    earnings: Optional[pd.DataFrame],
    news: Optional[pd.DataFrame],
    analyst: Optional[pd.DataFrame] = None,
    financials: Optional[pd.DataFrame] = None,
    macro_treasury: Optional[pd.DataFrame] = None,
    macro_calendar: Optional[pd.DataFrame] = None,
    adjusted_flag: bool,
    lookbacks: Sequence[int],
    event_safe_shift_days: int,
    news_general_symbol_uses_tickers: bool = False,
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
        n_feats = _prepare_news_features(
            news,
            event_safe_shift_days=event_safe_shift_days,
            general_symbol_uses_tickers=bool(news_general_symbol_uses_tickers),
        )
        topic_symbol_cols = [f"news_topic_{k}_1d" for k in _NEWS_TOPIC_KEYS]
        topic_market_cols = [f"mkt_news_topic_{k}_1d" for k in _NEWS_TOPIC_KEYS]
        if not n_feats.empty:
            sym_cols = [
                "news_count_1d",
                "news_sent_mean_1d",
                "news_source_nuniq_1d",
            ] + topic_symbol_cols
            mkt_cols = [
                "mkt_news_count_1d",
                "mkt_news_sent_mean_1d",
            ] + topic_market_cols
            sym_df = n_feats[n_feats["symbol"] != "GENERAL"][["decision_date", "symbol"] + sym_cols].copy()
            if not sym_df.empty:
                sym_df = sym_df.drop_duplicates(subset=["decision_date", "symbol"], keep="last")
                feats = feats.merge(sym_df, on=["decision_date", "symbol"], how="left")

            mkt_df = n_feats[["decision_date"] + mkt_cols].drop_duplicates(subset=["decision_date"], keep="last")
            if not mkt_df.empty:
                feats = feats.merge(mkt_df, on="decision_date", how="left")

        base_cols = [
            "news_count_1d",
            "news_sent_mean_1d",
            "news_source_nuniq_1d",
            "mkt_news_count_1d",
            "mkt_news_sent_mean_1d",
        ] + topic_symbol_cols + topic_market_cols
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
        for k in _NEWS_TOPIC_KEYS:
            sc = f"news_topic_{k}_1d"
            mc = f"mkt_news_topic_{k}_1d"
            s20 = f"news_topic_{k}_20d"
            m20 = f"mkt_news_topic_{k}_20d"
            feats[s20] = (
                feats.groupby("symbol")[sc].rolling(20, min_periods=1).sum().reset_index(level=0, drop=True)
            )
            feats[m20] = (
                feats.groupby("symbol")[mc].rolling(20, min_periods=1).sum().reset_index(level=0, drop=True)
            )

        for c in [
            "news_count_5d",
            "news_count_20d",
            "news_sent_mean_5d",
            "news_sent_std_20d",
            "mkt_news_count_20d",
            "mkt_news_sent_mean_5d",
            "news_attention_20d",
        ] + [f"news_topic_{k}_20d" for k in _NEWS_TOPIC_KEYS] + [f"mkt_news_topic_{k}_20d" for k in _NEWS_TOPIC_KEYS]:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)

    if analyst is not None:
        a_feats = _prepare_analyst_features(
            analyst,
            event_safe_shift_days=event_safe_shift_days,
        )
        feats = feats.merge(a_feats, on=["decision_date", "symbol"], how="left")
        analyst_cols = [c for c in a_feats.columns if c not in {"decision_date", "symbol"}]
        if analyst_cols:
            feats = feats.sort_values(["symbol", "decision_date"]).reset_index(drop=True)
            for c in analyst_cols:
                if c not in feats.columns:
                    feats[c] = np.nan
                feats[c] = pd.to_numeric(feats[c], errors="coerce")
                feats[c] = feats.groupby("symbol")[c].ffill().fillna(0.0)

    if macro_treasury is not None or macro_calendar is not None:
        t_feats = _prepare_macro_treasury_features(
            macro_treasury if macro_treasury is not None else pd.DataFrame(),
            event_safe_shift_days=event_safe_shift_days,
        )
        c_feats = _prepare_macro_calendar_features(
            macro_calendar if macro_calendar is not None else pd.DataFrame(),
            event_safe_shift_days=event_safe_shift_days,
        )

        if not t_feats.empty:
            feats = feats.merge(t_feats, on="decision_date", how="left")
        if not c_feats.empty:
            feats = feats.merge(c_feats, on="decision_date", how="left")

        treasury_cols = [c for c in t_feats.columns if c != "decision_date"]
        macro_event_cols = [c for c in c_feats.columns if c != "decision_date"]
        for c in treasury_cols + macro_event_cols:
            if c not in feats.columns:
                feats[c] = np.nan

        feats = feats.sort_values(["symbol", "decision_date"]).reset_index(drop=True)

        for c in treasury_cols:
            feats[c] = pd.to_numeric(feats[c], errors="coerce")
            feats[c] = feats.groupby("symbol")[c].ffill().bfill().fillna(0.0)

        for c in macro_event_cols:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)

        if "mkt_macro_events_us_1d" in feats.columns:
            feats["mkt_macro_events_us_5d"] = (
                feats.groupby("symbol")["mkt_macro_events_us_1d"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
            )
        if "mkt_macro_high_impact_us_1d" in feats.columns:
            feats["mkt_macro_high_impact_us_20d"] = (
                feats.groupby("symbol")["mkt_macro_high_impact_us_1d"].rolling(20, min_periods=1).sum().reset_index(level=0, drop=True)
            )
        if "mkt_macro_surprise_mean_1d" in feats.columns:
            feats["mkt_macro_surprise_mean_5d"] = (
                feats.groupby("symbol")["mkt_macro_surprise_mean_1d"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
            )
        if "mkt_macro_surprise_abs_mean_1d" in feats.columns:
            feats["mkt_macro_surprise_abs_mean_20d"] = (
                feats.groupby("symbol")["mkt_macro_surprise_abs_mean_1d"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
            )

        for c in [
            "mkt_macro_events_us_5d",
            "mkt_macro_high_impact_us_20d",
            "mkt_macro_surprise_mean_5d",
            "mkt_macro_surprise_abs_mean_20d",
        ]:
            if c not in feats.columns:
                feats[c] = 0.0
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)

    if financials is not None:
        f_feats = _prepare_financials_features(
            financials,
            event_safe_shift_days=event_safe_shift_days,
        )
        if not f_feats.empty:
            feats = feats.merge(f_feats, on=["decision_date", "symbol"], how="left")

        fin_cols = [c for c in f_feats.columns if c not in {"decision_date", "symbol"}]
        if fin_cols:
            feats = feats.sort_values(["symbol", "decision_date"]).reset_index(drop=True)
            for c in fin_cols:
                if c not in feats.columns:
                    feats[c] = np.nan
                feats[c] = pd.to_numeric(feats[c], errors="coerce")
                # Financial statements are event-driven snapshots; only forward-fill after publication.
                feats[c] = feats.groupby("symbol")[c].ffill().fillna(0.0)

    # Build spec + IDs
    feature_cols = [c for c in feats.columns if c not in {"decision_date", "symbol", "feature_available_date"}]
    spec = {
        "dataset_id": dataset_id,
        "lookbacks": list(lookbacks),
        "event_safe_shift_days": int(event_safe_shift_days),
        "include_news_features": bool(news is not None),
        "include_analyst_features": bool(analyst is not None),
        "include_financials_features": bool(financials is not None),
        "include_macro_features": bool(macro_treasury is not None or macro_calendar is not None),
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
