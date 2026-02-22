from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _resolve_fetch_workers(cfg: Dict[str, Any]) -> int:
    dcfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    raw = dcfg.get("fetch_workers", 1) if isinstance(dcfg, dict) else 1
    try:
        workers = int(raw)
    except Exception:
        workers = 1
    return max(1, workers)


def _parallel_symbol_fetch(
    symbols: Sequence[str],
    *,
    desc: str,
    max_workers: int,
    fetch_one: Callable[[str], Any],
) -> List[Tuple[str, Any, Optional[BaseException]]]:
    symbols_list = list(symbols)
    if not symbols_list:
        return []

    results: List[Optional[Tuple[str, Any, Optional[BaseException]]]] = [None] * len(symbols_list)
    if max_workers <= 1 or len(symbols_list) <= 1:
        for i, sym in enumerate(tqdm(symbols_list, desc=desc)):
            try:
                results[i] = (sym, fetch_one(sym), None)
            except Exception as exc:  # pragma: no cover - depends on remote behavior
                results[i] = (sym, None, exc)
        return [x for x in results if x is not None]

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{desc}_w") as pool:
        futures = {pool.submit(fetch_one, sym): i for i, sym in enumerate(symbols_list)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i = futures[fut]
            sym = symbols_list[i]
            try:
                payload = fut.result()
                results[i] = (sym, payload, None)
            except Exception as exc:  # pragma: no cover - depends on remote behavior
                results[i] = (sym, None, exc)

    return [x for x in results if x is not None]


def compute_dataset_id(
    cfg: Dict[str, Any],
    symbols: Sequence[str],
    *,
    effective_end_date: Optional[str] = None,
) -> str:
    dcfg = cfg.get("data", {}) or {}
    ucfg = cfg.get("universe", {}) or {}
    include_news = bool(dcfg.get("include_news", False))
    include_macro = bool(dcfg.get("include_macro", True))
    include_analyst = bool(dcfg.get("include_analyst", False))
    spec = {
        "universe_name": str(ucfg.get("name", "sp500_pit")),
        "symbols_hash": symbols_hash(list(symbols)),
        "start_date": str(dcfg.get("start_date")),
        "end_date": str(effective_end_date if effective_end_date is not None else (dcfg.get("end_date") or "")),
        "adjusted_flag": bool(dcfg.get("adjusted_flag", True)),
        "include_news": include_news,
        "general_news_page_size": int(dcfg.get("general_news_page_size", 200)) if include_news else 0,
        "general_news_max_pages": int(dcfg.get("general_news_max_pages", 1200)) if include_news else 0,
        "include_macro": include_macro,
        "include_analyst": include_analyst,
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
    fetch_workers = _resolve_fetch_workers(cfg)
    logger.info("data_fetch_workers=%d", fetch_workers)
    endpoints_version = [str(x).strip() for x in (dcfg.get("endpoints_version") or []) if str(x).strip()]
    enabled_endpoints = set(endpoints_version)

    def _endpoint_enabled(endpoint_name: str) -> bool:
        # Backward compatibility: if list is empty, treat all endpoints as enabled.
        if not enabled_endpoints:
            return True
        if endpoint_name in enabled_endpoints:
            return True
        # Backward-compatible aliases used by older configs.
        if endpoint_name in {"income_statement", "balance_sheet", "cash_flow"} and "financials" in enabled_endpoints:
            return True
        if endpoint_name in {
            "analyst_estimates",
            "grades",
            "grades_historical",
            "grades_consensus",
            "price_target_summary",
            "price_target_consensus",
            "ratings_historical",
        } and "analyst_premium" in enabled_endpoints:
            return True
        return endpoint_name in enabled_endpoints

    prices_path = raw_dataset_dir / "prices.parquet"
    dividends_path = raw_dataset_dir / "dividends.parquet"
    splits_path = raw_dataset_dir / "splits.parquet"
    earnings_path = raw_dataset_dir / "earnings.parquet"
    earnings_surprises_path = raw_dataset_dir / "earnings_surprises.parquet"
    financials_path = raw_dataset_dir / "financials_quarterly.parquet"
    news_path = raw_dataset_dir / "news.parquet"
    macro_treasury_path = raw_dataset_dir / "macro_treasury.parquet"
    macro_calendar_path = raw_dataset_dir / "macro_calendar.parquet"
    analyst_path = raw_dataset_dir / "analyst_premium.parquet"

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
        def _fetch_prices_one(sym: str) -> Any:
            return fmp.get_prices(sym, fetch_from if fetch_from else None, end if end else None)

        for sym, payload, exc in _parallel_symbol_fetch(
            symbols,
            desc="fetch_prices",
            max_workers=fetch_workers,
            fetch_one=_fetch_prices_one,
        ):
            if exc is not None:
                logger.warning(f"price_fetch_failed symbol={sym}: {type(exc).__name__}: {exc}")
                continue
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
    skip_non_price_refresh = bool(skip_price_fetch and not force)

    # Dividends / Splits (best-effort, cached; force controls refetch).
    def _fetch_actions(fn, out_path: Path, desc: str) -> None:
        if skip_non_price_refresh and out_path.exists():
            logger.info(f"skip_non_price_refresh desc={desc} reason=no_new_business_day")
            return
        existing = _read_parquet(out_path) if out_path.exists() and not force else pd.DataFrame()
        rows: List[pd.DataFrame] = []

        def fetch_fn(sym: str) -> Any:
            return fn(sym, start, end)

        for sym, payload, exc in _parallel_symbol_fetch(
            symbols,
            desc=desc,
            max_workers=fetch_workers,
            fetch_one=fetch_fn,
        ):
            if exc is not None:
                logger.warning(f"{desc}_failed symbol={sym}: {type(exc).__name__}: {exc}")
                continue
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

    if _endpoint_enabled("dividends"):
        _fetch_actions(fmp.get_dividends, dividends_path, "fetch_dividends")
    else:
        logger.info("skip_endpoint desc=fetch_dividends endpoint=dividends")
    if _endpoint_enabled("splits"):
        _fetch_actions(fmp.get_splits, splits_path, "fetch_splits")
    else:
        logger.info("skip_endpoint desc=fetch_splits endpoint=splits")

    # Earnings / Financials / News are best-effort to keep the platform operable.
    def _fetch_symbol_list(fn, out_path: Path, desc: str, max_rows: Optional[int] = None) -> None:
        if skip_non_price_refresh and out_path.exists():
            logger.info(f"skip_non_price_refresh desc={desc} reason=no_new_business_day")
            return
        existing = _read_parquet(out_path) if out_path.exists() and not force else pd.DataFrame()
        rows: List[pd.DataFrame] = []
        for sym, payload, exc in _parallel_symbol_fetch(
            symbols,
            desc=desc,
            max_workers=fetch_workers,
            fetch_one=fn,
        ):
            if exc is not None:
                logger.warning(f"{desc}_failed symbol={sym}: {type(exc).__name__}: {exc}")
                continue
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

    if _endpoint_enabled("earnings"):
        try:
            _fetch_symbol_list(fmp.get_earnings, earnings_path, "fetch_earnings")
        except Exception as e:
            logger.warning(f"earnings_fetch_failed: {type(e).__name__}: {e}")
    else:
        logger.info("skip_endpoint desc=fetch_earnings endpoint=earnings")

    if _endpoint_enabled("earnings_surprises"):
        try:
            if skip_non_price_refresh and earnings_surprises_path.exists():
                logger.info("skip_non_price_refresh desc=fetch_earnings_surprises reason=no_new_business_day")
            else:
                probe_symbols = list(symbols[: min(3, len(symbols))])
                endpoint_available = True
                if probe_symbols:
                    probe_failures = 0
                    for sym in probe_symbols:
                        try:
                            fmp.get_earnings_surprises(sym)
                        except Exception as probe_exc:
                            probe_failures += 1
                            logger.warning(
                                "earnings_surprises_probe_failed symbol=%s: %s: %s",
                                sym,
                                type(probe_exc).__name__,
                                probe_exc,
                            )
                    if probe_failures >= len(probe_symbols):
                        endpoint_available = False
                if endpoint_available:
                    _fetch_symbol_list(fmp.get_earnings_surprises, earnings_surprises_path, "fetch_earnings_surprises")
                else:
                    logger.warning("skip_endpoint desc=fetch_earnings_surprises reason=probe_all_failed")
        except Exception as e:
            logger.warning(f"earnings_surprises_fetch_failed: {type(e).__name__}: {e}")
    else:
        logger.info("skip_endpoint desc=fetch_earnings_surprises endpoint=earnings_surprises")

    def _fetch_financials() -> None:
        if skip_non_price_refresh and financials_path.exists():
            logger.info("skip_non_price_refresh desc=fetch_financials_quarterly reason=no_new_business_day")
            return
        existing = _read_parquet(financials_path) if financials_path.exists() and not force else pd.DataFrame()
        rows: List[pd.DataFrame] = []

        def _fetch_financials_one(sym: str) -> List[pd.DataFrame]:
            local_rows: List[pd.DataFrame] = []
            for stmt_name, endpoint_name, fn in [
                ("income", "income_statement", fmp.get_income_statement),
                ("balance", "balance_sheet", fmp.get_balance_sheet),
                ("cashflow", "cash_flow", fmp.get_cash_flow),
            ]:
                if not _endpoint_enabled(endpoint_name):
                    continue
                payload = fn(sym, period="quarter", limit=40)
                if not isinstance(payload, list) or not payload:
                    continue
                df = pd.DataFrame(payload)
                if df.empty:
                    continue
                df["symbol"] = sym
                df["statement_type"] = stmt_name
                local_rows.append(df)
            return local_rows

        for sym, payload_rows, exc in _parallel_symbol_fetch(
            symbols,
            desc="fetch_financials_quarterly",
            max_workers=fetch_workers,
            fetch_one=_fetch_financials_one,
        ):
            if exc is not None:
                logger.warning(f"financials_fetch_failed symbol={sym}: {type(exc).__name__}: {exc}")
                continue
            if isinstance(payload_rows, list):
                rows.extend([x for x in payload_rows if isinstance(x, pd.DataFrame) and not x.empty])
        new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        all_df = pd.concat([existing, new], ignore_index=True) if (not existing.empty and not new.empty) else (existing if not existing.empty else new)
        if all_df.empty:
            return
        _write_parquet(all_df, financials_path)

    if any(_endpoint_enabled(x) for x in ("income_statement", "balance_sheet", "cash_flow")):
        try:
            _fetch_financials()
        except Exception as e:
            logger.warning(f"financials_fetch_failed: {type(e).__name__}: {e}")
    else:
        logger.info("skip_endpoint desc=fetch_financials_quarterly endpoint=income_statement|balance_sheet|cash_flow")

    include_analyst = bool(dcfg.get("include_analyst", False))
    if include_analyst:
        analyst_endpoint_names = (
            "analyst_estimates",
            "grades",
            "grades_historical",
            "grades_consensus",
            "price_target_summary",
            "price_target_consensus",
            "ratings_historical",
        )
        if not any(_endpoint_enabled(x) for x in analyst_endpoint_names):
            logger.info("skip_endpoint desc=fetch_analyst_premium endpoint=analyst_*")
        elif skip_non_price_refresh and analyst_path.exists():
            logger.info("skip_non_price_refresh desc=fetch_analyst_premium reason=no_new_business_day")
        else:
            try:
                existing = _read_parquet(analyst_path) if analyst_path.exists() and not force else pd.DataFrame()
                rows: List[pd.DataFrame] = []

                def _to_frame(payload: Any) -> pd.DataFrame:
                    if isinstance(payload, dict):
                        return pd.DataFrame([payload])
                    if isinstance(payload, list):
                        return pd.DataFrame(payload)
                    return pd.DataFrame()

                def _fetch_analyst_one(sym: str) -> List[pd.DataFrame]:
                    local_rows: List[pd.DataFrame] = []
                    calls: List[Tuple[str, str, Callable[[], Any]]] = [
                        ("analyst_estimates", "analyst_estimates", lambda: fmp.get_analyst_estimates(sym, period="quarter", limit=40)),
                        ("grades", "grades", lambda: fmp.get_grades(sym, limit=100)),
                        ("grades_historical", "grades_historical", lambda: fmp.get_grades_historical(sym, limit=200)),
                        ("grades_consensus", "grades_consensus", lambda: fmp.get_grades_consensus(sym)),
                        ("price_target_summary", "price_target_summary", lambda: fmp.get_price_target_summary(sym)),
                        ("price_target_consensus", "price_target_consensus", lambda: fmp.get_price_target_consensus(sym)),
                        ("ratings_historical", "ratings_historical", lambda: fmp.get_ratings_historical(sym, limit=200)),
                    ]
                    for endpoint_name, source_name, fn in calls:
                        if not _endpoint_enabled(endpoint_name):
                            continue
                        try:
                            payload = fn()
                        except Exception as exc:
                            logger.warning(
                                "analyst_fetch_failed source=%s symbol=%s err=%s:%s",
                                source_name,
                                sym,
                                type(exc).__name__,
                                exc,
                            )
                            continue
                        df = _to_frame(payload)
                        if df.empty:
                            continue
                        df["symbol"] = sym
                        df["source"] = source_name
                        local_rows.append(df)
                    return local_rows

                for sym, payload_rows, exc in _parallel_symbol_fetch(
                    symbols,
                    desc="fetch_analyst_premium",
                    max_workers=fetch_workers,
                    fetch_one=_fetch_analyst_one,
                ):
                    if exc is not None:
                        logger.warning(f"fetch_analyst_premium_failed symbol={sym}: {type(exc).__name__}: {exc}")
                        continue
                    if isinstance(payload_rows, list):
                        rows.extend([x for x in payload_rows if isinstance(x, pd.DataFrame) and not x.empty])

                new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
                all_df = (
                    pd.concat([existing, new], ignore_index=True)
                    if (not existing.empty and not new.empty)
                    else (existing if not existing.empty else new)
                )
                if not all_df.empty:
                    dedupe_cols = [
                        c
                        for c in [
                            "symbol",
                            "source",
                            "date",
                            "publishedDate",
                            "published_date",
                            "gradingCompany",
                            "newGrade",
                            "previousGrade",
                            "title",
                            "url",
                            "newsURL",
                        ]
                        if c in all_df.columns
                    ]
                    if dedupe_cols:
                        all_df = all_df.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
                    _write_parquet(all_df, analyst_path)
            except Exception as e:
                logger.warning(f"analyst_premium_fetch_failed: {type(e).__name__}: {e}")

    include_macro = bool(dcfg.get("include_macro", True))
    if include_macro:
        if _endpoint_enabled("treasury_rates"):
            if skip_non_price_refresh and macro_treasury_path.exists():
                logger.info("skip_non_price_refresh desc=fetch_treasury_rates reason=no_new_business_day")
            else:
                try:
                    existing = _read_parquet(macro_treasury_path) if macro_treasury_path.exists() and not force else pd.DataFrame()
                    payload = fmp.get_treasury_rates(start, end)
                    new = pd.DataFrame(payload) if isinstance(payload, list) else pd.DataFrame()
                    if not new.empty and "date" in new.columns:
                        new["date"] = pd.to_datetime(new["date"], errors="coerce").dt.date
                        new = new[new["date"].notna()].reset_index(drop=True)
                        start_date = _parse_date(start)
                        end_date = _parse_date(end)
                        if start_date is not None:
                            new = new[new["date"] >= start_date]
                        if end_date is not None:
                            new = new[new["date"] <= end_date]
                    all_df = (
                        pd.concat([existing, new], ignore_index=True)
                        if (not existing.empty and not new.empty)
                        else (existing if not existing.empty else new)
                    )
                    if not all_df.empty and "date" in all_df.columns:
                        all_df = all_df.drop_duplicates(subset=["date"], keep="last").sort_values(["date"]).reset_index(drop=True)
                        _write_parquet(all_df, macro_treasury_path)
                except Exception as e:
                    logger.warning(f"macro_treasury_fetch_failed: {type(e).__name__}: {e}")
        else:
            logger.info("skip_endpoint desc=fetch_treasury_rates endpoint=treasury_rates")

        if _endpoint_enabled("macro_calendar"):
            if skip_non_price_refresh and macro_calendar_path.exists():
                logger.info("skip_non_price_refresh desc=fetch_macro_calendar reason=no_new_business_day")
            else:
                try:
                    existing = _read_parquet(macro_calendar_path) if macro_calendar_path.exists() and not force else pd.DataFrame()
                    payload = fmp.get_macro_calendar(start, end)
                    new = pd.DataFrame(payload) if isinstance(payload, list) else pd.DataFrame()
                    if not new.empty and "date" in new.columns:
                        ts = pd.to_datetime(new["date"], errors="coerce", utc=True)
                        new["_event_date"] = ts.dt.tz_localize(None).dt.date
                        new = new[new["_event_date"].notna()].reset_index(drop=True)
                        start_date = _parse_date(start)
                        end_date = _parse_date(end)
                        if start_date is not None:
                            new = new[new["_event_date"] >= start_date]
                        if end_date is not None:
                            new = new[new["_event_date"] <= end_date]
                        new = new.drop(columns=["_event_date"])
                    all_df = (
                        pd.concat([existing, new], ignore_index=True)
                        if (not existing.empty and not new.empty)
                        else (existing if not existing.empty else new)
                    )
                    if not all_df.empty:
                        dedupe_cols = [c for c in ["date", "country", "event", "currency", "actual", "estimate", "previous"] if c in all_df.columns]
                        if dedupe_cols:
                            all_df = all_df.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
                        _write_parquet(all_df, macro_calendar_path)
                except Exception as e:
                    logger.warning(f"macro_calendar_fetch_failed: {type(e).__name__}: {e}")
        else:
            logger.info("skip_endpoint desc=fetch_macro_calendar endpoint=macro_calendar")

    include_news = bool(dcfg.get("include_news", False))
    if include_news:
        news_stock_enabled = _endpoint_enabled("stock_news")
        news_general_enabled = _endpoint_enabled("general_news")
        if not news_stock_enabled and not news_general_enabled:
            logger.info("skip_endpoint desc=fetch_news endpoint=stock_news|general_news")
            include_news = False
    if include_news:
        if skip_non_price_refresh and news_path.exists():
            logger.info("skip_non_price_refresh desc=fetch_news reason=no_new_business_day")
        else:
            try:
                existing = _read_parquet(news_path) if news_path.exists() and not force else pd.DataFrame()
                rows: List[pd.DataFrame] = []
                stock_failures = 0
                stock_max_failures = 5
                start_date = _parse_date(start)
                end_date = _parse_date(end)

                def _news_event_dates(frame: pd.DataFrame) -> pd.Series:
                    for cand in ("publishedDate", "published_date", "date"):
                        if cand in frame.columns:
                            ts = pd.to_datetime(frame[cand], errors="coerce", utc=True)
                            return ts.dt.tz_localize(None).dt.date
                    return pd.Series([pd.NaT] * len(frame), index=frame.index)

                if news_stock_enabled:
                    for sym in tqdm(symbols, desc="fetch_news"):
                        try:
                            payload = fmp.get_stock_news(sym, limit=50)
                        except Exception as e:
                            stock_failures += 1
                            logger.warning(f"stock_news_fetch_failed symbol={sym}: {type(e).__name__}: {e}")
                            # Avoid long retry tails when endpoint is unavailable for the current key/tier.
                            if stock_failures >= stock_max_failures and not rows:
                                logger.warning("stock_news_fetch_disabled after repeated failures; trying general_news only")
                                break
                            continue
                        if not isinstance(payload, list) or not payload:
                            continue
                        df = pd.DataFrame(payload)
                        if df.empty:
                            continue
                        df["symbol"] = sym
                        rows.append(df)
                else:
                    logger.info("skip_endpoint desc=fetch_news endpoint=stock_news")

                if news_general_enabled:
                    try:
                        page_size = max(1, min(int(dcfg.get("general_news_page_size", 200)), 200))
                        max_pages = max(1, int(dcfg.get("general_news_max_pages", 1200)))

                        existing_min_date: Optional[date] = None
                        existing_covers_start = False
                        if not existing.empty:
                            existing_dates = _news_event_dates(existing)
                            if existing_dates.notna().any():
                                existing_min_date = existing_dates.dropna().min()
                                if start_date is not None and existing_min_date <= start_date:
                                    existing_covers_start = True

                        seen_page_markers: set[str] = set()
                        repeated_page_count = 0

                        def _general_news_page_marker(frame: pd.DataFrame) -> Optional[str]:
                            if frame.empty:
                                return None
                            marker_cols = [
                                c
                                for c in (
                                    "publishedDate",
                                    "published_date",
                                    "date",
                                    "title",
                                    "url",
                                    "link",
                                    "symbol",
                                    "tickers",
                                    "ticker",
                                )
                                if c in frame.columns
                            ]
                            if not marker_cols:
                                return None
                            marker_rows = (
                                frame[marker_cols]
                                .fillna("")
                                .astype(str)
                                .agg("|".join, axis=1)
                                .tolist()
                            )
                            return sha1_hex("\n".join(marker_rows))

                        for page in range(max_pages):
                            gnews = fmp.get_general_news(limit=page_size, page=page)
                            if not isinstance(gnews, list) or not gnews:
                                break
                            gdf = pd.DataFrame(gnews)
                            if gdf.empty:
                                break
                            gdf["_event_date"] = _news_event_dates(gdf)
                            gdf = gdf[gdf["_event_date"].notna()].copy()
                            if gdf.empty:
                                continue

                            oldest = gdf["_event_date"].min()
                            page_marker = _general_news_page_marker(gdf)
                            if page_marker in seen_page_markers:
                                repeated_page_count += 1
                                # Some tiers return repeated pages for large page ranges; stop after streak.
                                if repeated_page_count >= 3:
                                    break
                                continue
                            repeated_page_count = 0
                            if page_marker:
                                seen_page_markers.add(page_marker)

                            if end_date is not None:
                                gdf = gdf[gdf["_event_date"] <= end_date]
                            if start_date is not None:
                                gdf = gdf[gdf["_event_date"] >= start_date]
                            if not gdf.empty:
                                if "symbol" not in gdf.columns:
                                    gdf["symbol"] = "GENERAL"
                                rows.append(gdf.drop(columns=["_event_date"]))

                            if start_date is not None and oldest <= start_date:
                                break
                            if existing_covers_start and existing_min_date is not None and oldest <= existing_min_date:
                                break
                    except Exception as e:
                        logger.warning(f"general_news_fetch_failed: {type(e).__name__}: {e}")
                else:
                    logger.info("skip_endpoint desc=fetch_news endpoint=general_news")

                new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
                all_df = (
                    pd.concat([existing, new], ignore_index=True)
                    if (not existing.empty and not new.empty)
                    else (existing if not existing.empty else new)
                )
                if not all_df.empty:
                    dedupe_cols = [c for c in ["symbol", "publishedDate", "date", "title", "url", "link"] if c in all_df.columns]
                    if dedupe_cols:
                        all_df = all_df.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
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
        "include_news": bool(dcfg.get("include_news", False)),
        "general_news_page_size": int(dcfg.get("general_news_page_size", 200)) if bool(dcfg.get("include_news", False)) else 0,
        "general_news_max_pages": int(dcfg.get("general_news_max_pages", 1200)) if bool(dcfg.get("include_news", False)) else 0,
        "include_macro": bool(dcfg.get("include_macro", True)),
        "include_analyst": bool(dcfg.get("include_analyst", False)),
        "endpoints_version": list(dcfg.get("endpoints_version", [])),
        "timezone_assumption": str(cfg.get("run", {}).get("timezone_assumption", "")),
    }
    (raw_dataset_dir / "dataset_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=True))

    return dataset_id, universe, last_data_date
