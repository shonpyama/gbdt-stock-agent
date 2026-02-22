from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.features import build_feature_store


def _make_prices(symbols: list[str], dates: list) -> pd.DataFrame:
    rows: list[dict] = []
    for j, sym in enumerate(symbols):
        px = 100.0 + float(j)
        for i, d in enumerate(dates):
            px = px * (1.0 + 0.001 + (i % 5) * 0.0002)
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": px * 0.999,
                    "high": px * 1.002,
                    "low": px * 0.998,
                    "close": px,
                    "adj_close": px,
                    "volume": 100000 + (i % 20) * 1000,
                }
            )
    return pd.DataFrame(rows)


def _make_membership(symbols: list[str], dates: list) -> pd.DataFrame:
    return pd.DataFrame([{"date": d, "symbol": s, "is_member": True} for d in dates for s in symbols])


def test_build_feature_store_with_news_adds_news_columns(tmp_path: Path) -> None:
    symbols = ["AAA", "BBB"]
    dates = list(pd.bdate_range("2024-01-01", "2024-06-30").date)
    prices = _make_prices(symbols, dates)
    membership = _make_membership(symbols, dates)

    earnings = pd.DataFrame(columns=["symbol", "date"])
    news = pd.DataFrame(
        [
            {"publishedDate": str(dates[40]), "symbol": "AAA", "title": "Strong growth beats estimates", "site": "siteA"},
            {"publishedDate": str(dates[41]), "symbol": "AAA", "title": "Upgrade and upside outlook", "site": "siteB"},
            {"publishedDate": str(dates[45]), "symbol": "BBB", "title": "Warning on weak demand", "site": "siteA"},
            {"publishedDate": str(dates[46]), "symbol": "GENERAL", "title": "Market rally after policy update", "site": "macro"},
        ]
    )

    res = build_feature_store(
        dataset_id="ds_news",
        prices=prices,
        universe_membership=membership,
        earnings=earnings,
        news=news,
        adjusted_flag=True,
        lookbacks=[1, 5, 20, 60],
        event_safe_shift_days=1,
        out_dir=tmp_path / "feature_store",
    )

    cols = set(res.features.columns)
    assert "news_count_1d" in cols
    assert "news_count_5d" in cols
    assert "news_count_20d" in cols
    assert "mkt_news_count_20d" in cols
    assert "news_attention_20d" in cols
    assert bool((res.features["news_count_20d"] > 0.0).any())


def test_build_feature_store_without_news_keeps_legacy_feature_set(tmp_path: Path) -> None:
    symbols = ["AAA", "BBB"]
    dates = list(pd.bdate_range("2024-01-01", "2024-06-30").date)
    prices = _make_prices(symbols, dates)
    membership = _make_membership(symbols, dates)
    earnings = pd.DataFrame(columns=["symbol", "date"])

    res = build_feature_store(
        dataset_id="ds_no_news",
        prices=prices,
        universe_membership=membership,
        earnings=earnings,
        news=None,
        adjusted_flag=True,
        lookbacks=[1, 5, 20, 60],
        event_safe_shift_days=1,
        out_dir=tmp_path / "feature_store",
    )

    news_cols = [c for c in res.features.columns if c.startswith("news_") or c.startswith("mkt_news_")]
    assert news_cols == []


def test_build_feature_store_with_general_news_tickers_maps_multiple_symbols(tmp_path: Path) -> None:
    symbols = ["AAA", "BBB", "CCC"]
    dates = list(pd.bdate_range("2024-01-01", "2024-06-30").date)
    prices = _make_prices(symbols, dates)
    membership = _make_membership(symbols, dates)
    earnings = pd.DataFrame(columns=["symbol", "date"])

    news = pd.DataFrame(
        [
            {
                "date": str(dates[40]),
                "tickers": "NASDAQ:BBB,NYSE:CCC",
                "title": "Upgrade after strong growth outlook",
                "content": "Analysts upgrade name and raise estimates",
                "site": "fmp",
            }
        ]
    )

    res = build_feature_store(
        dataset_id="ds_tickers_news",
        prices=prices,
        universe_membership=membership,
        earnings=earnings,
        news=news,
        adjusted_flag=True,
        lookbacks=[1, 5, 20, 60],
        event_safe_shift_days=1,
        out_dir=tmp_path / "feature_store",
    )

    bbb = res.features[res.features["symbol"] == "BBB"]
    ccc = res.features[res.features["symbol"] == "CCC"]
    aaa = res.features[res.features["symbol"] == "AAA"]
    assert bool((bbb["news_count_20d"] > 0.0).any())
    assert bool((ccc["news_count_20d"] > 0.0).any())
    assert not bool((aaa["news_count_20d"] > 0.0).any())


def test_build_feature_store_market_news_applies_to_all_symbols_same_day(tmp_path: Path) -> None:
    symbols = ["AAA", "BBB", "CCC"]
    dates = list(pd.bdate_range("2024-01-01", "2024-06-30").date)
    prices = _make_prices(symbols, dates)
    membership = _make_membership(symbols, dates)
    earnings = pd.DataFrame(columns=["symbol", "date"])

    news = pd.DataFrame(
        [
            {
                "date": str(dates[40]),
                "title": "Macro update lifts market risk appetite",
                "content": "Policy update improves broad sentiment",
                "site": "macro-wire",
            }
        ]
    )

    res = build_feature_store(
        dataset_id="ds_market_news",
        prices=prices,
        universe_membership=membership,
        earnings=earnings,
        news=news,
        adjusted_flag=True,
        lookbacks=[1, 5, 20, 60],
        event_safe_shift_days=1,
        out_dir=tmp_path / "feature_store",
    )

    decision_date = dates[41]
    day = res.features[res.features["decision_date"] == decision_date]
    assert len(day) == len(symbols)
    assert bool((day["mkt_news_count_1d"] > 0.0).all())
    assert bool((day["news_count_1d"] == 0.0).all())


def test_build_feature_store_general_symbol_falls_back_to_tickers(tmp_path: Path) -> None:
    symbols = ["AAA", "BBB"]
    dates = list(pd.bdate_range("2024-01-01", "2024-06-30").date)
    prices = _make_prices(symbols, dates)
    membership = _make_membership(symbols, dates)
    earnings = pd.DataFrame(columns=["symbol", "date"])

    news = pd.DataFrame(
        [
            {
                "date": str(dates[30]),
                "symbol": "GENERAL",
                "tickers": "NASDAQ:BBB",
                "title": "BBB upgrade",
                "content": "analyst upgrade",
                "site": "wire",
            }
        ]
    )

    res = build_feature_store(
        dataset_id="ds_general_symbol_tickers_fallback",
        prices=prices,
        universe_membership=membership,
        earnings=earnings,
        news=news,
        adjusted_flag=True,
        lookbacks=[1, 5, 20, 60],
        event_safe_shift_days=1,
        news_general_symbol_uses_tickers=True,
        out_dir=tmp_path / "feature_store",
    )

    bbb = res.features[res.features["symbol"] == "BBB"]
    aaa = res.features[res.features["symbol"] == "AAA"]
    assert bool((bbb["news_count_20d"] > 0.0).any())
    assert not bool((aaa["news_count_20d"] > 0.0).any())


def test_build_feature_store_general_symbol_keeps_market_only_by_default(tmp_path: Path) -> None:
    symbols = ["AAA", "BBB"]
    dates = list(pd.bdate_range("2024-01-01", "2024-06-30").date)
    prices = _make_prices(symbols, dates)
    membership = _make_membership(symbols, dates)
    earnings = pd.DataFrame(columns=["symbol", "date"])

    news = pd.DataFrame(
        [
            {
                "date": str(dates[30]),
                "symbol": "GENERAL",
                "tickers": "NASDAQ:BBB",
                "title": "BBB upgrade",
                "content": "analyst upgrade",
                "site": "wire",
            }
        ]
    )

    res = build_feature_store(
        dataset_id="ds_general_symbol_default",
        prices=prices,
        universe_membership=membership,
        earnings=earnings,
        news=news,
        adjusted_flag=True,
        lookbacks=[1, 5, 20, 60],
        event_safe_shift_days=1,
        out_dir=tmp_path / "feature_store",
    )

    day = res.features[res.features["decision_date"] == dates[31]]
    assert len(day) == len(symbols)
    assert bool((day["mkt_news_count_1d"] > 0.0).all())
    assert bool((day["news_count_1d"] == 0.0).all())


def test_build_feature_store_news_dates_coalesce_published_date_and_date(tmp_path: Path) -> None:
    symbols = ["AAA", "BBB"]
    dates = list(pd.bdate_range("2024-01-01", "2024-06-30").date)
    prices = _make_prices(symbols, dates)
    membership = _make_membership(symbols, dates)
    earnings = pd.DataFrame(columns=["symbol", "date"])

    news = pd.DataFrame(
        [
            {
                "publishedDate": f"{dates[20]} 12:00:00",
                "date": None,
                "tickers": "NASDAQ:AAA",
                "title": "Earnings beat expectations",
                "site": "wire-a",
            },
            {
                "publishedDate": None,
                "date": f"{dates[40]} 12:00:00",
                "tickers": "NASDAQ:AAA",
                "title": "Analyst upgrade",
                "site": "wire-b",
            },
        ]
    )

    res = build_feature_store(
        dataset_id="ds_news_date_coalesce",
        prices=prices,
        universe_membership=membership,
        earnings=earnings,
        news=news,
        adjusted_flag=True,
        lookbacks=[1, 5, 20, 60],
        event_safe_shift_days=1,
        out_dir=tmp_path / "feature_store",
    )

    d1 = dates[21]
    d2 = dates[41]
    row1 = res.features[(res.features["symbol"] == "AAA") & (res.features["decision_date"] == d1)]
    row2 = res.features[(res.features["symbol"] == "AAA") & (res.features["decision_date"] == d2)]
    assert len(row1) == 1
    assert len(row2) == 1
    assert float(row1.iloc[0]["news_count_1d"]) > 0.0
    assert float(row2.iloc[0]["news_count_1d"]) > 0.0
