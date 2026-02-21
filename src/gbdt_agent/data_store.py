from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.tseries.offsets import BDay

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

from .config import sha1_hex, stable_json_dumps, symbols_hash
from .fmp_client import FMPClient
from .paths import ProjectPaths


logger = logging.getLogger(__name__)
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def _ensure_date_str(value: Optional[str | date]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _today_utc_date() -> date:
    return datetime.now(timezone.utc).date()


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    return ts.date()


def dataset_id_from_spec(spec: Dict[str, Any]) -> str:
    payload = stable_json_dumps(spec)
    return sha1_hex(payload)[:12]


@dataclass(frozen=True)
class UniverseResult:
    provider_used: str
    fallback_used: bool
    fallback_reason: str
    end_date_truncated: bool
    start_date: str
    end_date_effective: str
    membership_long: pd.DataFrame  # columns: date, symbol, is_member
    symbols_union: List[str]


def load_fallback_small50(conf_dir: Path) -> List[str]:
    path = conf_dir / "universe_sp500.yaml"
    if not path.exists():
        return []
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    symbols = raw.get("fallback_small50", {}).get("symbols", []) or []
    return [str(s).strip().upper() for s in symbols if str(s).strip()]


def load_custom_universe(conf_dir: Path) -> List[str]:
    path = conf_dir / "universe_custom.yaml"
    if not path.exists():
        return []
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    symbols = raw.get("symbols", []) or []
    return [str(s).strip().upper() for s in symbols if str(s).strip()]


def _extract_symbol(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().upper()
        return s if (s and _TICKER_RE.match(s)) else None
    if isinstance(value, dict):
        for k in ("symbol", "ticker", "Symbol", "Ticker"):
            v = value.get(k)
            if isinstance(v, str) and v.strip():
                s = v.strip().upper()
                if _TICKER_RE.match(s):
                    return s
    return None


def _parse_sp500_current_symbols(payload: Any) -> List[str]:
    if not isinstance(payload, list):
        return []
    out: List[str] = []
    for row in payload:
        sym = _extract_symbol(row.get("symbol") if isinstance(row, dict) else None) if isinstance(row, dict) else None
        if sym:
            out.append(sym)
    return sorted(set(out))


@dataclass(frozen=True)
class Sp500Event:
    event_date: date
    added: Optional[str]
    removed: Optional[str]


def _parse_sp500_events(payload: Any) -> List[Sp500Event]:
    if not isinstance(payload, list):
        return []

    added_keys = (
        "addedSymbol",
        "added_symbol",
        "addedSecuritySymbol",
        "symbolAdded",
        "addedTicker",
        "added",
        "addedSecurity",
        "newSymbol",
    )
    removed_keys = (
        "removedSymbol",
        "removed_symbol",
        "removedSecuritySymbol",
        "symbolRemoved",
        "removedTicker",
        "removed",
        "removedSecurity",
        "oldSymbol",
    )

    events: List[Sp500Event] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        d = _parse_date(row.get("date") or row.get("effectiveDate") or row.get("eventDate"))
        if d is None:
            continue

        added = None
        removed = None
        for k in added_keys:
            if k in row:
                added = _extract_symbol(row.get(k))
                if added:
                    break
        for k in removed_keys:
            if k in row:
                removed = _extract_symbol(row.get(k))
                if removed:
                    break
        events.append(Sp500Event(event_date=d, added=added, removed=removed))
    events.sort(key=lambda e: e.event_date)
    return events


def _next_business_day(d: date) -> date:
    return (pd.Timestamp(d) + BDay(1)).date()


def build_sp500_point_in_time_membership(
    fmp: FMPClient,
    start_date: date,
    end_date: date,
) -> Tuple[pd.DataFrame, List[str], bool]:
    """
    Returns:
      membership_long: date, symbol, is_member
      union_symbols
      end_date_truncated (if end_date was truncated to today)
    """
    today = _today_utc_date()
    end_truncated = False
    if end_date > today:
        end_date = today
        end_truncated = True

    current = fmp.get_sp500_constituent()
    current_symbols = _parse_sp500_current_symbols(current)
    if not current_symbols:
        raise RuntimeError("Failed to parse /sp500-constituent symbols")

    hist = fmp.get_historical_sp500_constituent()
    events = _parse_sp500_events(hist)

    # Business-day schedule (approximate trading days).
    bdays = list(pd.bdate_range(start_date, end_date).date)
    if not bdays:
        raise RuntimeError("No business days in requested range")
    bday_set = set(bdays)

    # Map events to business days (if event date is not a business day, shift to next business day).
    events_by_date: Dict[date, List[Sp500Event]] = {}
    for e in events:
        if e.event_date < start_date or e.event_date > end_date:
            continue
        d_eff = e.event_date if e.event_date in bday_set else _next_business_day(e.event_date)
        if d_eff < start_date or d_eff > end_date:
            continue
        events_by_date.setdefault(d_eff, []).append(e)

    # Compute membership at end_date by rolling back events after end_date.
    set_at_end = set(current_symbols)
    for e in reversed(events):
        if e.event_date > end_date:
            if e.added and e.added in set_at_end:
                set_at_end.remove(e.added)
            if e.removed:
                set_at_end.add(e.removed)

    # Compute membership just before start_date changes by rolling back events within [start_date, end_date].
    set_prev = set(set_at_end)
    for e in reversed(events):
        if start_date <= e.event_date <= end_date:
            if e.added and e.added in set_prev:
                set_prev.remove(e.added)
            if e.removed:
                set_prev.add(e.removed)

    current_set = set_prev
    membership_records: List[Dict[str, Any]] = []
    for d in bdays:
        for e in events_by_date.get(d, []):
            if e.added:
                current_set.add(e.added)
            if e.removed and e.removed in current_set:
                current_set.remove(e.removed)
        for sym in current_set:
            membership_records.append({"date": d, "symbol": sym, "is_member": True})

    membership_long = pd.DataFrame(membership_records)
    union_symbols = sorted(membership_long["symbol"].unique().tolist()) if not membership_long.empty else []
    return membership_long, union_symbols, end_truncated


def resolve_universe(
    cfg: Dict[str, Any],
    paths: ProjectPaths,
    fmp: FMPClient,
) -> UniverseResult:
    ucfg = cfg.get("universe", {}) or {}
    provider = str(ucfg.get("provider", "sp500_point_in_time"))
    fallback_cfg = ucfg.get("fallback_small50", {}) or {}
    fallback_enabled = bool(fallback_cfg.get("enabled", True))

    start = _parse_date(cfg.get("data", {}).get("start_date"))
    end_raw = cfg.get("data", {}).get("end_date")
    end = _parse_date(end_raw) or _today_utc_date()
    if start is None:
        raise ValueError("data.start_date is required")

    if provider == "custom_list":
        syms = load_custom_universe(paths.conf_dir)
        if not syms:
            raise RuntimeError("universe.provider=custom_list but conf/universe_custom.yaml has no symbols")
        bdays = list(pd.bdate_range(start, end).date)
        membership_long = pd.DataFrame([{"date": d, "symbol": s, "is_member": True} for d in bdays for s in syms])
        return UniverseResult(
            provider_used="custom_list",
            fallback_used=False,
            fallback_reason="",
            end_date_truncated=False,
            start_date=start.isoformat(),
            end_date_effective=end.isoformat(),
            membership_long=membership_long,
            symbols_union=sorted(set(syms)),
        )

    # sp500_point_in_time
    try:
        membership_long, union_syms, end_trunc = build_sp500_point_in_time_membership(fmp, start, end)
        if not union_syms:
            raise RuntimeError("Empty PIT S&P500 universe")
        return UniverseResult(
            provider_used="sp500_point_in_time",
            fallback_used=False,
            fallback_reason="",
            end_date_truncated=end_trunc,
            start_date=start.isoformat(),
            end_date_effective=min(end, _today_utc_date()).isoformat(),
            membership_long=membership_long,
            symbols_union=union_syms,
        )
    except Exception as e:
        if not fallback_enabled:
            raise
        fb_syms = (fallback_cfg.get("symbols") or []) and [str(s).strip().upper() for s in fallback_cfg.get("symbols") or []]
        if not fb_syms:
            fb_syms = load_fallback_small50(paths.conf_dir)
        if not fb_syms:
            raise RuntimeError(f"Universe fallback enabled but no fallback symbols available: {type(e).__name__}: {e}")
        logger.warning(f"Universe fallback_small50 used due to: {type(e).__name__}: {e}")
        bdays = list(pd.bdate_range(start, end).date)
        membership_long = pd.DataFrame([{"date": d, "symbol": s, "is_member": True} for d in bdays for s in fb_syms])
        return UniverseResult(
            provider_used="fallback_small50",
            fallback_used=True,
            fallback_reason=f"{type(e).__name__}: {e}",
            end_date_truncated=end > _today_utc_date(),
            start_date=start.isoformat(),
            end_date_effective=min(end, _today_utc_date()).isoformat(),
            membership_long=membership_long,
            symbols_union=sorted(set(fb_syms)),
        )


def _to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def compute_dataset_id(
    cfg: Dict[str, Any],
    symbols: Sequence[str],
    *,
    effective_end_date: Optional[str] = None,
) -> str:
    dcfg = cfg.get("data", {}) or {}
    ucfg = cfg.get("universe", {}) or {}
    spec = {
        "universe_name": str(ucfg.get("name", "sp500_pit")),
        "symbols_hash": symbols_hash(list(symbols)),
        "start_date": str(dcfg.get("start_date")),
        "end_date": str(effective_end_date if effective_end_date is not None else (dcfg.get("end_date") or "")),
        "adjusted_flag": bool(dcfg.get("adjusted_flag", True)),
        "endpoints_version": list(dcfg.get("endpoints_version", [])),
        "timezone_assumption": str(cfg.get("run", {}).get("timezone_assumption", "")),
    }
    return dataset_id_from_spec(spec)


def update_data(
    cfg: Dict[str, Any],
    paths: ProjectPaths,
    fmp: FMPClient,
    *,
    force: bool = False,
) -> Tuple[str, UniverseResult, str]:
    """
    Fetches raw data and writes into data/raw/<dataset_id>/.
    Returns: (dataset_id, universe_result, last_data_date_iso)
    """
    universe = resolve_universe(cfg, paths, fmp)
    symbols = universe.symbols_union

    dataset_id = compute_dataset_id(cfg, symbols, effective_end_date=universe.end_date_effective)
    raw_dataset_dir = paths.raw_dir / dataset_id
    raw_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save universe membership for reproducibility & PIT usage.
    _write_parquet(universe.membership_long, raw_dataset_dir / "universe_membership.parquet")

    dcfg = cfg.get("data", {}) or {}
    start = str(dcfg.get("start_date"))
    end = str(universe.end_date_effective)

    prices_path = raw_dataset_dir / "prices.parquet"
    dividends_path = raw_dataset_dir / "dividends.parquet"
    splits_path = raw_dataset_dir / "splits.parquet"
    earnings_path = raw_dataset_dir / "earnings.parquet"
    earnings_surprises_path = raw_dataset_dir / "earnings_surprises.parquet"
    financials_path = raw_dataset_dir / "financials_quarterly.parquet"
    news_path = raw_dataset_dir / "news.parquet"

    # Prices (incremental by global max date; per-symbol gaps are tolerated).
    prices_existing = _read_parquet(prices_path) if prices_path.exists() and not force else pd.DataFrame()
    last_date_existing: Optional[date] = None
    if not prices_existing.empty:
        prices_existing = _to_datetime(prices_existing, "date")
        last_date_existing = prices_existing["date"].max().date()

    fetch_from = start
    fetch_end = end
    skip_price_fetch = False
    if last_date_existing is not None and str(last_date_existing) >= start:
        # Incremental refresh starts from the next business day after existing max date.
        fetch_from_date = (pd.Timestamp(last_date_existing) + BDay(1)).date()
        fetch_end_date = pd.to_datetime(fetch_end).date()
        if fetch_from_date > fetch_end_date:
            skip_price_fetch = True
        else:
            fetch_from = fetch_from_date.isoformat()

    price_rows: List[pd.DataFrame] = []
    if skip_price_fetch:
        logger.info(f"skip_price_incremental_fetch no_new_business_day fetch_from>{fetch_end}")
    else:
        for sym in tqdm(symbols, desc="fetch_prices"):
            payload = fmp.get_prices(sym, fetch_from if fetch_from else None, end if end else None)
            if not isinstance(payload, list):
                continue
            df = pd.DataFrame(payload)
            if df.empty:
                continue
            df["symbol"] = sym
            # Normalize columns
            rename = {}
            if "adjClose" in df.columns:
                rename["adjClose"] = "adj_close"
            if "unadjustedClose" in df.columns:
                rename["unadjustedClose"] = "unadjusted_close"
            df = df.rename(columns=rename)
            keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
            df = df[keep + ["symbol"]]
            df["date"] = pd.to_datetime(df["date"]).dt.date
            price_rows.append(df)

    prices_new = pd.concat(price_rows, ignore_index=True) if price_rows else pd.DataFrame()
    if not prices_existing.empty and not prices_new.empty:
        merged = pd.concat([prices_existing, prices_new], ignore_index=True)
    elif not prices_existing.empty:
        merged = prices_existing
    else:
        merged = prices_new

    if merged.empty:
        raise RuntimeError("No price data fetched")
    merged = merged.drop_duplicates(subset=["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    _write_parquet(merged, prices_path)

    last_data_date = str(pd.to_datetime(merged["date"]).max().date())

    # Dividends / Splits (best-effort, cached; force controls refetch).
    def _fetch_actions(fn, out_path: Path, desc: str) -> None:
        existing = _read_parquet(out_path) if out_path.exists() and not force else pd.DataFrame()
        rows: List[pd.DataFrame] = []
        for sym in tqdm(symbols, desc=desc):
            payload = fn(sym, start, end)
            if not isinstance(payload, list) or not payload:
                continue
            df = pd.DataFrame(payload)
            if df.empty:
                continue
            df["symbol"] = sym
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            rows.append(df)
        new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not existing.empty and not new.empty:
            all_df = pd.concat([existing, new], ignore_index=True)
        else:
            all_df = existing if not existing.empty else new
        if all_df.empty:
            return
        if "date" in all_df.columns:
            all_df = all_df.drop_duplicates(subset=["date", "symbol"], keep="last")
        _write_parquet(all_df, out_path)

    _fetch_actions(fmp.get_dividends, dividends_path, "fetch_dividends")
    _fetch_actions(fmp.get_splits, splits_path, "fetch_splits")

    # Earnings / Financials / News are best-effort to keep the platform operable.
    def _fetch_symbol_list(fn, out_path: Path, desc: str, max_rows: Optional[int] = None) -> None:
        existing = _read_parquet(out_path) if out_path.exists() and not force else pd.DataFrame()
        rows: List[pd.DataFrame] = []
        for sym in tqdm(symbols, desc=desc):
            payload = fn(sym)
            if not isinstance(payload, list) or not payload:
                continue
            df = pd.DataFrame(payload)
            if df.empty:
                continue
            df["symbol"] = sym
            rows.append(df)
        new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if max_rows is not None and not new.empty and len(new) > max_rows:
            new = new.head(max_rows)
        all_df = pd.concat([existing, new], ignore_index=True) if (not existing.empty and not new.empty) else (existing if not existing.empty else new)
        if all_df.empty:
            return
        _write_parquet(all_df, out_path)

    try:
        _fetch_symbol_list(fmp.get_earnings, earnings_path, "fetch_earnings")
    except Exception as e:
        logger.warning(f"earnings_fetch_failed: {type(e).__name__}: {e}")

    try:
        _fetch_symbol_list(fmp.get_earnings_surprises, earnings_surprises_path, "fetch_earnings_surprises")
    except Exception as e:
        logger.warning(f"earnings_surprises_fetch_failed: {type(e).__name__}: {e}")

    def _fetch_financials() -> None:
        existing = _read_parquet(financials_path) if financials_path.exists() and not force else pd.DataFrame()
        rows: List[pd.DataFrame] = []
        for sym in tqdm(symbols, desc="fetch_financials_quarterly"):
            for stmt_name, fn in [
                ("income", fmp.get_income_statement),
                ("balance", fmp.get_balance_sheet),
                ("cashflow", fmp.get_cash_flow),
            ]:
                payload = fn(sym, period="quarter", limit=40)
                if not isinstance(payload, list) or not payload:
                    continue
                df = pd.DataFrame(payload)
                if df.empty:
                    continue
                df["symbol"] = sym
                df["statement_type"] = stmt_name
                rows.append(df)
        new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        all_df = pd.concat([existing, new], ignore_index=True) if (not existing.empty and not new.empty) else (existing if not existing.empty else new)
        if all_df.empty:
            return
        _write_parquet(all_df, financials_path)

    try:
        _fetch_financials()
    except Exception as e:
        logger.warning(f"financials_fetch_failed: {type(e).__name__}: {e}")

    include_news = bool(dcfg.get("include_news", False))
    if include_news:
        try:
            existing = _read_parquet(news_path) if news_path.exists() and not force else pd.DataFrame()
            rows: List[pd.DataFrame] = []
            for sym in tqdm(symbols, desc="fetch_news"):
                payload = fmp.get_stock_news(sym, limit=50)
                if not isinstance(payload, list) or not payload:
                    continue
                df = pd.DataFrame(payload)
                if df.empty:
                    continue
                df["symbol"] = sym
                rows.append(df)
            try:
                gnews = fmp.get_general_news(limit=200)
                if isinstance(gnews, list) and gnews:
                    gdf = pd.DataFrame(gnews)
                    if not gdf.empty:
                        if "symbol" not in gdf.columns:
                            gdf["symbol"] = "GENERAL"
                        rows.append(gdf)
            except Exception as e:
                logger.warning(f"general_news_fetch_failed: {type(e).__name__}: {e}")
            new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            all_df = pd.concat([existing, new], ignore_index=True) if (not existing.empty and not new.empty) else (existing if not existing.empty else new)
            if not all_df.empty:
                _write_parquet(all_df, news_path)
        except Exception as e:
            logger.warning(f"news_fetch_failed: {type(e).__name__}: {e}")

    # Save dataset spec for reproducibility.
    spec = {
        "dataset_id": dataset_id,
        "provider_used": universe.provider_used,
        "fallback_used": universe.fallback_used,
        "fallback_reason": universe.fallback_reason,
        "end_date_truncated": universe.end_date_truncated,
        "start_date": universe.start_date,
        "end_date_effective": universe.end_date_effective,
        "symbols_count": len(symbols),
        "symbols_hash": symbols_hash(symbols),
        "adjusted_flag": bool(dcfg.get("adjusted_flag", True)),
        "endpoints_version": list(dcfg.get("endpoints_version", [])),
        "timezone_assumption": str(cfg.get("run", {}).get("timezone_assumption", "")),
    }
    (raw_dataset_dir / "dataset_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=True))

    return dataset_id, universe, last_data_date
