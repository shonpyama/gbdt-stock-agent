from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.fmp_client import FMPClient, FMPClientConfig


def test_default_news_endpoint_mapping(tmp_path: Path) -> None:
    client = FMPClient(FMPClientConfig(api_key="dummy"), cache_dir=tmp_path)
    assert client.endpoint_for("stock_news") == "news/stock"
    assert client.endpoint_for("general_news") == "fmp-articles"


def test_get_stock_news_uses_short_retry_and_symbol_params(monkeypatch, tmp_path: Path) -> None:
    client = FMPClient(FMPClientConfig(api_key="dummy"), cache_dir=tmp_path)
    captured = {}

    def _fake_request(endpoint, params=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["endpoint"] = endpoint
        captured["params"] = dict(params or {})
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(client, "request", _fake_request)
    client.get_stock_news("AAPL", limit=12)

    assert captured["endpoint"] == "news/stock"
    assert captured["params"] == {"symbol": "AAPL", "limit": 12}
    assert captured["kwargs"]["endpoint_name"] == "stock_news"
    assert captured["kwargs"]["max_attempts"] == 2


def test_get_general_news_uses_fmp_articles_page_and_size(monkeypatch, tmp_path: Path) -> None:
    client = FMPClient(FMPClientConfig(api_key="dummy"), cache_dir=tmp_path)
    captured = {}

    def _fake_request(endpoint, params=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["endpoint"] = endpoint
        captured["params"] = dict(params or {})
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(client, "request", _fake_request)
    client.get_general_news(limit=250)

    assert captured["endpoint"] == "fmp-articles"
    assert captured["params"] == {"page": 0, "size": 200}
    assert captured["kwargs"]["endpoint_name"] == "general_news"
    assert captured["kwargs"]["max_attempts"] == 2

