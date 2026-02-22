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
