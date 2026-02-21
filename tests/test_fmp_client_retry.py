from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gbdt_agent.fmp_client as fmp_client
from gbdt_agent.fmp_client import FMPClient, FMPClientConfig


class _AlwaysFailSession:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise RuntimeError("boom")


def test_request_respects_max_attempts_override(monkeypatch, tmp_path: Path) -> None:
    cfg = FMPClientConfig(api_key="dummy", retry_max_attempts=6, retry_backoff_base_seconds=0.01)
    client = FMPClient(cfg=cfg, cache_dir=tmp_path)
    session = _AlwaysFailSession()
    client.session = session
    client.rate_limiter.acquire = lambda: None  # type: ignore[method-assign]
    monkeypatch.setattr(fmp_client.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError):
        client.request("earnings-surprises", params={"symbol": "A"}, endpoint_name="earnings_surprises", force=True, max_attempts=1)
    assert session.calls == 1


def test_request_uses_config_retry_by_default(monkeypatch, tmp_path: Path) -> None:
    cfg = FMPClientConfig(api_key="dummy", retry_max_attempts=3, retry_backoff_base_seconds=0.01)
    client = FMPClient(cfg=cfg, cache_dir=tmp_path)
    session = _AlwaysFailSession()
    client.session = session
    client.rate_limiter.acquire = lambda: None  # type: ignore[method-assign]
    monkeypatch.setattr(fmp_client.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError):
        client.request("historical-price-eod/full", params={"symbol": "A"}, endpoint_name="prices_eod", force=True)
    assert session.calls == 3
