from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.data_store import _extract_symbol, _parse_date, compute_dataset_id, resolve_universe, update_data
from gbdt_agent.paths import ProjectPaths


def test_parse_date_rejects_empty_and_invalid_values() -> None:
    assert _parse_date("") is None
    assert _parse_date("   ") is None
    assert _parse_date("not-a-date") is None
    assert _parse_date("2026-02-21") == date(2026, 2, 21)


def test_extract_symbol_rejects_company_names() -> None:
    assert _extract_symbol("AAPL") == "AAPL"
    assert _extract_symbol("BRK.B") == "BRK.B"
    assert _extract_symbol("BLOCK, INC.") is None
    assert _extract_symbol("AIRBNB INC") is None


def test_resolve_universe_uses_today_when_end_date_is_blank(monkeypatch, tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(
        yaml.safe_dump({"symbols": ["AAPL", "MSFT"]}, sort_keys=False)
    )

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-17",
            "end_date": "",
        },
    }
    monkeypatch.setattr("gbdt_agent.data_store._today_utc_date", lambda: date(2026, 2, 21))
    paths = ProjectPaths.from_project_dir(tmp_path)

    result = resolve_universe(cfg, paths, fmp=object())
    assert result.end_date_effective == "2026-02-21"
    assert result.start_date == "2026-02-17"
    assert result.membership_long.empty is False


def test_update_data_skips_price_fetch_when_no_new_business_day(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-20",
            "end_date": "2026-02-21",  # Saturday
            "adjusted_flag": False,
            "endpoints_version": ["prices_eod"],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    dsid = compute_dataset_id(cfg, ["AAPL"], effective_end_date="2026-02-21")
    raw = paths.raw_dir / dsid
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": date(2026, 2, 20),
                "symbol": "AAPL",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100.0,
            }
        ]
    ).to_parquet(raw / "prices.parquet", index=False)

    class DummyFMP:
        def __init__(self) -> None:
            self.price_calls = 0

        def get_prices(self, *args, **kwargs):
            self.price_calls += 1
            return []

        def get_dividends(self, *args, **kwargs):
            return []

        def get_splits(self, *args, **kwargs):
            return []

        def get_earnings(self, *args, **kwargs):
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

    fmp = DummyFMP()
    _, _, last_data_date = update_data(cfg, paths, fmp, force=False)
    assert fmp.price_calls == 0
    assert last_data_date == "2026-02-20"


def test_update_data_skips_non_price_refresh_when_no_new_business_day(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-20",
            "end_date": "2026-02-21",  # Saturday
            "adjusted_flag": False,
            "include_news": True,
            "include_macro": True,
            "endpoints_version": [
                "prices_eod",
                "dividends",
                "splits",
                "earnings",
                "earnings_surprises",
                "financials",
                "stock_news",
                "general_news",
                "treasury_rates",
                "macro_calendar",
            ],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    dsid = compute_dataset_id(cfg, ["AAPL"], effective_end_date="2026-02-21")
    raw = paths.raw_dir / dsid
    raw.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "date": date(2026, 2, 20),
                "symbol": "AAPL",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100.0,
            }
        ]
    ).to_parquet(raw / "prices.parquet", index=False)

    # Existing artifacts should prevent non-price refresh calls when no new business day exists.
    for fname in [
        "dividends.parquet",
        "splits.parquet",
        "earnings.parquet",
        "earnings_surprises.parquet",
        "financials_quarterly.parquet",
        "news.parquet",
        "macro_treasury.parquet",
        "macro_calendar.parquet",
    ]:
        pd.DataFrame({"date": [date(2026, 2, 20)], "symbol": ["AAPL"]}).to_parquet(raw / fname, index=False)

    class DummyFMP:
        def __init__(self) -> None:
            self.calls = {
                "prices": 0,
                "dividends": 0,
                "splits": 0,
                "earnings": 0,
                "earnings_surprises": 0,
                "income": 0,
                "balance": 0,
                "cashflow": 0,
                "stock_news": 0,
                "general_news": 0,
                "treasury_rates": 0,
                "macro_calendar": 0,
            }

        def get_prices(self, *args, **kwargs):
            self.calls["prices"] += 1
            return []

        def get_dividends(self, *args, **kwargs):
            self.calls["dividends"] += 1
            return []

        def get_splits(self, *args, **kwargs):
            self.calls["splits"] += 1
            return []

        def get_earnings(self, *args, **kwargs):
            self.calls["earnings"] += 1
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            self.calls["earnings_surprises"] += 1
            return []

        def get_income_statement(self, *args, **kwargs):
            self.calls["income"] += 1
            return []

        def get_balance_sheet(self, *args, **kwargs):
            self.calls["balance"] += 1
            return []

        def get_cash_flow(self, *args, **kwargs):
            self.calls["cashflow"] += 1
            return []

        def get_stock_news(self, *args, **kwargs):
            self.calls["stock_news"] += 1
            return []

        def get_general_news(self, *args, **kwargs):
            self.calls["general_news"] += 1
            return []

        def get_treasury_rates(self, *args, **kwargs):
            self.calls["treasury_rates"] += 1
            return []

        def get_macro_calendar(self, *args, **kwargs):
            self.calls["macro_calendar"] += 1
            return []

    fmp = DummyFMP()
    _, _, last_data_date = update_data(cfg, paths, fmp, force=False)

    assert last_data_date == "2026-02-20"
    assert fmp.calls["prices"] == 0
    assert fmp.calls["dividends"] == 0
    assert fmp.calls["splits"] == 0
    assert fmp.calls["earnings"] == 0
    assert fmp.calls["earnings_surprises"] == 0
    assert fmp.calls["income"] == 0
    assert fmp.calls["balance"] == 0
    assert fmp.calls["cashflow"] == 0
    assert fmp.calls["stock_news"] == 0
    assert fmp.calls["general_news"] == 0
    assert fmp.calls["treasury_rates"] == 0
    assert fmp.calls["macro_calendar"] == 0


def test_update_data_parallel_price_fetch_workers(tmp_path: Path) -> None:
    import threading

    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN"]
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": symbols}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-19",
            "end_date": "2026-02-20",
            "adjusted_flag": False,
            "fetch_workers": 4,
            "include_news": False,
            "include_macro": False,
            "endpoints_version": ["prices_eod"],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    class DummyFMP:
        def __init__(self) -> None:
            self.price_calls = 0
            self._lock = threading.Lock()

        def get_prices(self, symbol: str, *args, **kwargs):
            with self._lock:
                self.price_calls += 1
            return [
                {
                    "date": "2026-02-20",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 1000.0,
                }
            ]

        def get_dividends(self, *args, **kwargs):
            return []

        def get_splits(self, *args, **kwargs):
            return []

        def get_earnings(self, *args, **kwargs):
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

    fmp = DummyFMP()
    dataset_id, _, _ = update_data(cfg, paths, fmp, force=False)
    prices_path = paths.raw_dir / dataset_id / "prices.parquet"
    prices = pd.read_parquet(prices_path)

    assert fmp.price_calls == len(symbols)
    assert sorted(prices["symbol"].astype(str).unique().tolist()) == sorted(symbols)
    assert len(prices) == len(symbols)


def test_update_data_keeps_general_news_when_stock_news_fails(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-20",
            "end_date": "2026-02-20",
            "adjusted_flag": False,
            "include_news": True,
            "endpoints_version": ["prices_eod", "stock_news", "general_news"],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    class DummyFMP:
        def get_prices(self, *args, **kwargs):
            return [
                {
                    "date": "2026-02-20",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 1000.0,
                }
            ]

        def get_dividends(self, *args, **kwargs):
            return []

        def get_splits(self, *args, **kwargs):
            return []

        def get_earnings(self, *args, **kwargs):
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

        def get_stock_news(self, *args, **kwargs):
            raise RuntimeError("stock news unavailable")

        def get_general_news(self, *args, **kwargs):
            return [
                {
                    "title": "General market update",
                    "date": "2026-02-20 12:00:00",
                    "content": "macro trend",
                    "tickers": "NASDAQ:AAPL",
                    "site": "fmp",
                }
            ]

    fmp = DummyFMP()
    dataset_id, _, _ = update_data(cfg, paths, fmp, force=False)
    news_path = paths.raw_dir / dataset_id / "news.parquet"
    assert news_path.exists()
    news_df = pd.read_parquet(news_path)
    assert len(news_df) == 1
    assert news_df.iloc[0]["tickers"] == "NASDAQ:AAPL"


def test_update_data_general_news_paginates_until_start_date(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-18",
            "end_date": "2026-02-21",
            "adjusted_flag": False,
            "include_news": True,
            "general_news_page_size": 20,
            "general_news_max_pages": 10,
            "endpoints_version": ["prices_eod", "stock_news", "general_news"],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    class DummyFMP:
        def __init__(self) -> None:
            self.general_pages: list[int] = []

        def get_prices(self, *args, **kwargs):
            return [
                {
                    "date": "2026-02-20",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 1000.0,
                }
            ]

        def get_dividends(self, *args, **kwargs):
            return []

        def get_splits(self, *args, **kwargs):
            return []

        def get_earnings(self, *args, **kwargs):
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

        def get_stock_news(self, *args, **kwargs):
            return []

        def get_general_news(self, *, limit: int = 100, page: int = 0):
            self.general_pages.append(int(page))
            pages = {
                0: [{"title": "t0", "date": "2026-02-21 12:00:00", "tickers": "NASDAQ:AAPL"}],
                1: [{"title": "t1", "date": "2026-02-19 12:00:00", "tickers": "NASDAQ:AAPL"}],
                2: [{"title": "t2", "date": "2026-02-18 12:00:00", "tickers": "NASDAQ:AAPL"}],
                3: [{"title": "t3", "date": "2026-02-17 12:00:00", "tickers": "NASDAQ:AAPL"}],
            }
            return pages.get(int(page), [])

    fmp = DummyFMP()
    dataset_id, _, _ = update_data(cfg, paths, fmp, force=False)
    news_path = paths.raw_dir / dataset_id / "news.parquet"
    news_df = pd.read_parquet(news_path)

    assert fmp.general_pages == [0, 1, 2]
    assert len(news_df) == 3
    dts = pd.to_datetime(news_df["date"], errors="coerce").dt.date
    assert dts.min() == date(2026, 2, 18)


def test_update_data_general_news_allows_same_date_range_with_different_pages(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-19",
            "end_date": "2026-02-21",
            "adjusted_flag": False,
            "include_news": True,
            "general_news_page_size": 20,
            "general_news_max_pages": 10,
            "endpoints_version": ["prices_eod", "stock_news", "general_news"],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    class DummyFMP:
        def __init__(self) -> None:
            self.general_pages: list[int] = []

        def get_prices(self, *args, **kwargs):
            return [
                {
                    "date": "2026-02-20",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 1000.0,
                }
            ]

        def get_dividends(self, *args, **kwargs):
            return []

        def get_splits(self, *args, **kwargs):
            return []

        def get_earnings(self, *args, **kwargs):
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

        def get_stock_news(self, *args, **kwargs):
            return []

        def get_general_news(self, *, limit: int = 100, page: int = 0):
            self.general_pages.append(int(page))
            pages = {
                0: [{"title": "t0", "date": "2026-02-20 12:00:00", "tickers": "NASDAQ:AAPL"}],
                1: [{"title": "t1", "date": "2026-02-20 12:00:00", "tickers": "NASDAQ:AAPL"}],
                2: [{"title": "t2", "date": "2026-02-19 12:00:00", "tickers": "NASDAQ:AAPL"}],
                3: [],
            }
            return pages.get(int(page), [])

    fmp = DummyFMP()
    dataset_id, _, _ = update_data(cfg, paths, fmp, force=False)
    news_path = paths.raw_dir / dataset_id / "news.parquet"
    news_df = pd.read_parquet(news_path)

    assert fmp.general_pages == [0, 1, 2]
    assert len(news_df) == 3
    assert set(news_df["title"].astype(str).tolist()) == {"t0", "t1", "t2"}


def test_update_data_fetches_macro_files_when_enabled(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-19",
            "end_date": "2026-02-21",
            "adjusted_flag": False,
            "include_news": False,
            "include_macro": True,
            "endpoints_version": ["prices_eod", "treasury_rates", "macro_calendar"],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    class DummyFMP:
        def get_prices(self, *args, **kwargs):
            return [
                {
                    "date": "2026-02-20",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 1000.0,
                }
            ]

        def get_dividends(self, *args, **kwargs):
            return []

        def get_splits(self, *args, **kwargs):
            return []

        def get_earnings(self, *args, **kwargs):
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

        def get_treasury_rates(self, start=None, end=None):
            return [
                {"date": "2026-02-18", "month3": 3.8, "year2": 3.6, "year10": 4.1},
                {"date": "2026-02-20", "month3": 3.7, "year2": 3.5, "year10": 4.0},
            ]

        def get_macro_calendar(self, start=None, end=None):
            return [
                {
                    "date": "2026-02-20 13:30:00",
                    "country": "US",
                    "event": "CPI (MoM)",
                    "impact": "High",
                    "actual": "0.4%",
                    "estimate": "0.3%",
                }
            ]

    fmp = DummyFMP()
    dataset_id, _, _ = update_data(cfg, paths, fmp, force=False)

    treasury_path = paths.raw_dir / dataset_id / "macro_treasury.parquet"
    calendar_path = paths.raw_dir / dataset_id / "macro_calendar.parquet"
    assert treasury_path.exists()
    assert calendar_path.exists()

    tdf = pd.read_parquet(treasury_path)
    cdf = pd.read_parquet(calendar_path)
    assert len(tdf) == 1
    assert pd.to_datetime(tdf.iloc[0]["date"]).date() == date(2026, 2, 20)
    assert len(cdf) == 1
    assert str(cdf.iloc[0]["country"]).upper() == "US"


def test_update_data_skips_earnings_surprises_when_endpoint_not_enabled(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-19",
            "end_date": "2026-02-20",
            "adjusted_flag": False,
            "include_news": False,
            "include_macro": False,
            "include_analyst": False,
            "endpoints_version": ["prices_eod", "earnings"],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    class DummyFMP:
        def __init__(self) -> None:
            self.calls = {
                "prices": 0,
                "earnings": 0,
                "earnings_surprises": 0,
                "dividends": 0,
                "splits": 0,
            }

        def get_prices(self, *args, **kwargs):
            self.calls["prices"] += 1
            return [
                {
                    "date": "2026-02-20",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 1000.0,
                }
            ]

        def get_earnings(self, *args, **kwargs):
            self.calls["earnings"] += 1
            return [{"date": "2026-02-20", "eps": 1.23}]

        def get_earnings_surprises(self, *args, **kwargs):
            self.calls["earnings_surprises"] += 1
            return [{"date": "2026-02-20", "actualEarningResult": 1.2}]

        def get_dividends(self, *args, **kwargs):
            self.calls["dividends"] += 1
            return []

        def get_splits(self, *args, **kwargs):
            self.calls["splits"] += 1
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

    fmp = DummyFMP()
    dataset_id, _, _ = update_data(cfg, paths, fmp, force=False)
    raw_dir = paths.raw_dir / dataset_id

    assert fmp.calls["prices"] == 1
    assert fmp.calls["earnings"] == 1
    assert fmp.calls["earnings_surprises"] == 0
    assert fmp.calls["dividends"] == 0
    assert fmp.calls["splits"] == 0
    assert (raw_dir / "earnings.parquet").exists()
    assert not (raw_dir / "earnings_surprises.parquet").exists()


def test_update_data_respects_endpoints_version_for_analyst_sources(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-19",
            "end_date": "2026-02-20",
            "adjusted_flag": False,
            "include_news": False,
            "include_macro": False,
            "include_analyst": True,
            "endpoints_version": ["prices_eod", "grades"],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    class DummyFMP:
        def __init__(self) -> None:
            self.calls = {
                "grades": 0,
                "analyst_estimates": 0,
                "grades_historical": 0,
                "grades_consensus": 0,
                "price_target_summary": 0,
                "price_target_consensus": 0,
                "ratings_historical": 0,
            }

        def get_prices(self, *args, **kwargs):
            return [
                {
                    "date": "2026-02-20",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 1000.0,
                }
            ]

        def get_dividends(self, *args, **kwargs):
            return []

        def get_splits(self, *args, **kwargs):
            return []

        def get_earnings(self, *args, **kwargs):
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

        def get_analyst_estimates(self, *args, **kwargs):
            self.calls["analyst_estimates"] += 1
            return []

        def get_grades(self, *args, **kwargs):
            self.calls["grades"] += 1
            return [{"date": "2026-02-20", "gradingCompany": "X", "newGrade": "Buy", "previousGrade": "Hold"}]

        def get_grades_historical(self, *args, **kwargs):
            self.calls["grades_historical"] += 1
            return []

        def get_grades_consensus(self, *args, **kwargs):
            self.calls["grades_consensus"] += 1
            return []

        def get_price_target_summary(self, *args, **kwargs):
            self.calls["price_target_summary"] += 1
            return []

        def get_price_target_consensus(self, *args, **kwargs):
            self.calls["price_target_consensus"] += 1
            return []

        def get_ratings_historical(self, *args, **kwargs):
            self.calls["ratings_historical"] += 1
            return []

    fmp = DummyFMP()
    dataset_id, _, _ = update_data(cfg, paths, fmp, force=False)
    analyst_path = paths.raw_dir / dataset_id / "analyst_premium.parquet"

    assert fmp.calls["grades"] == 1
    assert fmp.calls["analyst_estimates"] == 0
    assert fmp.calls["grades_historical"] == 0
    assert fmp.calls["grades_consensus"] == 0
    assert fmp.calls["price_target_summary"] == 0
    assert fmp.calls["price_target_consensus"] == 0
    assert fmp.calls["ratings_historical"] == 0
    assert analyst_path.exists()

    adf = pd.read_parquet(analyst_path)
    assert set(adf["source"].astype(str).unique().tolist()) == {"grades"}
