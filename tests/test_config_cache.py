from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.data_store import compute_dataset_id
from gbdt_agent.fmp_client import cache_key


def test_dataset_id_is_reproducible_and_symbol_order_independent() -> None:
    cfg = {
        "universe": {"name": "sp500_pit"},
        "data": {
            "start_date": "2024-01-01",
            "end_date": "",
            "adjusted_flag": True,
            "endpoints_version": ["prices_eod", "splits"],
        },
        "run": {"timezone_assumption": "US/Eastern;events_effective=next_day"},
    }
    a = compute_dataset_id(cfg, ["MSFT", "AAPL", "NVDA"], effective_end_date="2024-12-31")
    b = compute_dataset_id(cfg, ["NVDA", "MSFT", "AAPL"], effective_end_date="2024-12-31")
    c = compute_dataset_id(cfg, ["NVDA", "MSFT", "AAPL"], effective_end_date="2025-01-01")
    assert a == b
    assert a != c


def test_cache_key_does_not_depend_on_apikey() -> None:
    k1 = cache_key(
        "GET",
        "https://financialmodelingprep.com/stable/historical-price-eod/full",
        {"symbol": "AAPL", "from": "2024-01-01", "to": "2024-01-31", "apikey": "SECRET_A"},
    )
    k2 = cache_key(
        "GET",
        "https://financialmodelingprep.com/stable/historical-price-eod/full",
        {"symbol": "AAPL", "from": "2024-01-01", "to": "2024-01-31", "apikey": "SECRET_B"},
    )
    assert k1 == k2
