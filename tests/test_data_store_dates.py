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
