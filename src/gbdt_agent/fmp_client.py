from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlencode

import requests


logger = logging.getLogger(__name__)


API_KEY_PARAM = "apikey"
DEFAULT_API_KEY_FILES = [Path("/content/.env_fmp")]

DEFAULT_ENDPOINTS = {
    "sp500_constituent": "sp500-constituent",
    "historical_sp500_constituent": "historical-sp500-constituent",
    "prices_eod": "historical-price-eod/full",
    "dividends": "dividends",
    "splits": "splits",
    "earnings": "earnings",
    "earnings_calendar": "earnings-calendar",
    "earnings_surprises": "earnings-surprises",
    "income_statement": "income-statement",
    "balance_sheet": "balance-sheet-statement",
    "cash_flow": "cash-flow-statement",
    "stock_news": "stock-news",
    "general_news": "general-news",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_query_params(params: Mapping[str, Any]) -> str:
    items = []
    for k in sorted(params.keys()):
        if str(k).lower() == API_KEY_PARAM:
            continue
        v = params[k]
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            for vv in v:
                items.append((k, str(vv)))
        else:
            items.append((k, str(v)))
    return urlencode(items, doseq=True)


def cache_key(method: str, url_path: str, params: Mapping[str, Any]) -> str:
    canon = _canonical_query_params(params)
    blob = f"{method.upper()}|{url_path}|{canon}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _cache_file(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _strip_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and ((v[0] == "'" and v[-1] == "'") or (v[0] == '"' and v[-1] == '"')):
        return v[1:-1].strip()
    return v


def _parse_api_key_line(line: str) -> Optional[str]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("export "):
        s = s[len("export ") :].strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", s)
    if m:
        name = m.group(1)
        value = m.group(2).strip()
        if name != "FMP_API_KEY":
            return None
        quoted = re.match(r"""^(['"])(.*?)\1(?:\s*#.*)?$""", value)
        if quoted:
            return quoted.group(2).strip() or None
        # For unquoted values, allow trailing inline comments.
        value = value.split("#", 1)[0].strip()
        return _strip_quotes(value) or None
    # Plain key file support: accept a bare token line only.
    if "=" in s:
        return None
    if any(ch.isspace() for ch in s):
        return None
    return _strip_quotes(s) or None


def _load_api_key_from_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    try:
        for raw in path.read_text().splitlines():
            key = _parse_api_key_line(raw)
            if key:
                return key
    except Exception:
        return None
    return None


def resolve_fmp_api_key() -> Optional[str]:
    import os

    env_key = os.environ.get("FMP_API_KEY")
    if env_key:
        return env_key

    candidates: list[Path] = []
    key_file_env = os.environ.get("FMP_API_KEY_FILE")
    if key_file_env:
        candidates.append(Path(key_file_env).expanduser())
    candidates.extend(DEFAULT_API_KEY_FILES)

    for p in candidates:
        key = _load_api_key_from_file(p)
        if key:
            return key
    return None


class RateLimiter:
    def __init__(self, max_calls_per_minute: int):
        self.max_calls_per_minute = max(1, int(max_calls_per_minute))
        self._calls: list[float] = []

    def acquire(self) -> None:
        now = time.monotonic()
        window = 60.0
        while True:
            self._calls = [t for t in self._calls if (now - t) < window]
            if len(self._calls) < self.max_calls_per_minute:
                self._calls.append(now)
                return
            sleep_for = window - (now - min(self._calls))
            time.sleep(max(0.01, sleep_for))
            now = time.monotonic()


@dataclass(frozen=True)
class FMPClientConfig:
    api_key: str
    base_url: str = "https://financialmodelingprep.com/stable"
    timeout: int = 30
    max_calls_per_minute: int = 300
    retry_max_attempts: int = 6
    retry_backoff_base_seconds: float = 1.0
    endpoints: Dict[str, str] | None = None


class FMPClient:
    def __init__(self, cfg: FMPClientConfig, cache_dir: Path):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(cfg.max_calls_per_minute)
        self.endpoints = dict(DEFAULT_ENDPOINTS)
        if cfg.endpoints:
            self.endpoints.update({str(k): str(v) for k, v in cfg.endpoints.items()})

    @staticmethod
    def from_env(cache_dir: Path, cfg_overrides: Optional[Dict[str, Any]] = None) -> "FMPClient":
        api_key = resolve_fmp_api_key()
        if not api_key:
            raise RuntimeError("Missing FMP_API_KEY (env var or /content/.env_fmp)")
        cfg_overrides = cfg_overrides or {}
        rate = cfg_overrides.get("rate_limit", {}) if isinstance(cfg_overrides.get("rate_limit", {}), dict) else {}
        retry = cfg_overrides.get("retry", {}) if isinstance(cfg_overrides.get("retry", {}), dict) else {}
        cfg = FMPClientConfig(
            api_key=api_key,
            base_url=str(cfg_overrides.get("base_url", "https://financialmodelingprep.com/stable")),
            timeout=int(cfg_overrides.get("timeout", 30)),
            max_calls_per_minute=int(rate.get("max_calls_per_minute", cfg_overrides.get("max_calls_per_minute", 300))),
            retry_max_attempts=int(retry.get("max_attempts", cfg_overrides.get("retry_max_attempts", 6))),
            retry_backoff_base_seconds=float(retry.get("backoff_base_seconds", cfg_overrides.get("retry_backoff_base_seconds", 1.0))),
            endpoints=(cfg_overrides.get("endpoints") if isinstance(cfg_overrides.get("endpoints"), dict) else None),
        )
        return FMPClient(cfg=cfg, cache_dir=cache_dir)

    def endpoint_for(self, endpoint_name: str) -> str:
        raw = self.endpoints.get(endpoint_name, endpoint_name)
        return str(raw).lstrip("/")

    def _read_cache(self, key: str) -> Optional[Any]:
        path = _cache_file(self.cache_dir, key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return payload.get("data")

    def _write_cache(self, key: str, meta: Dict[str, Any], data: Any) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_file(self.cache_dir, key)
        payload = {"meta": meta, "data": data}
        path.write_text(json.dumps(payload, ensure_ascii=True))

    def request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        force: bool = False,
        endpoint_name: Optional[str] = None,
        max_attempts: Optional[int] = None,
    ) -> Any:
        endpoint = endpoint.lstrip("/")
        url_path = f"{self.cfg.base_url.rstrip('/')}/{endpoint}"
        params = dict(params or {})

        safe_params = {k: v for k, v in params.items() if str(k).lower() != API_KEY_PARAM}
        key = cache_key("GET", url_path, safe_params)

        if not force:
            cached = self._read_cache(key)
            if cached is not None:
                logger.info(f"cache_hit endpoint={endpoint_name or endpoint} cache_key={key}")
                return cached

        logger.info(f"cache_miss endpoint={endpoint_name or endpoint} cache_key={key} params={safe_params}")

        params[API_KEY_PARAM] = self.cfg.api_key
        self.rate_limiter.acquire()
        attempts_limit = max(1, int(max_attempts if max_attempts is not None else self.cfg.retry_max_attempts))

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self.session.get(url_path, params=params, timeout=self.cfg.timeout)
                status = resp.status_code
                if status == 200:
                    data = resp.json()
                    meta = {
                        "timestamp": _utc_now_iso(),
                        "status_code": status,
                        "endpoint_name": endpoint_name or endpoint,
                        "symbol": safe_params.get("symbol") or safe_params.get("ticker"),
                        "date_range": {"from": safe_params.get("from"), "to": safe_params.get("to")},
                        "cache_key": key,
                    }
                    self._write_cache(key, meta=meta, data=data)
                    return data

                retryable = status in (429, 500, 502, 503, 504)
                if not retryable or attempt >= attempts_limit:
                    body = None
                    try:
                        body = resp.text[:500]
                    except Exception:
                        body = "<unreadable>"
                    raise RuntimeError(f"HTTP {status} endpoint={endpoint} body={body}")

                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except Exception:
                        sleep_s = None
                else:
                    sleep_s = None
                if sleep_s is None:
                    base = self.cfg.retry_backoff_base_seconds
                    sleep_s = base * (2 ** (attempt - 1)) + random.random() * 0.1
                time.sleep(min(60.0, max(0.1, sleep_s)))
            except Exception as e:
                if attempt >= attempts_limit:
                    raise RuntimeError(f"request_failed endpoint={endpoint} err={type(e).__name__}") from e
                base = self.cfg.retry_backoff_base_seconds
                sleep_s = base * (2 ** (attempt - 1)) + random.random() * 0.1
                logger.warning(f"request_retry attempt={attempt} endpoint={endpoint} err={type(e).__name__}")
                time.sleep(min(60.0, max(0.1, sleep_s)))

    # --- Endpoints ---
    def get_sp500_constituent(self) -> Any:
        ep = self.endpoint_for("sp500_constituent")
        return self.request(ep, endpoint_name="sp500_constituent")

    def get_historical_sp500_constituent(self) -> Any:
        ep = self.endpoint_for("historical_sp500_constituent")
        return self.request(ep, endpoint_name="historical_sp500_constituent")

    def get_prices(self, symbol: str, start: Optional[str], end: Optional[str]) -> Any:
        params: Dict[str, Any] = {"symbol": symbol}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        ep = self.endpoint_for("prices_eod")
        return self.request(ep, params=params, endpoint_name="prices_eod")

    def get_dividends(self, symbol: str, start: Optional[str], end: Optional[str]) -> Any:
        params: Dict[str, Any] = {"symbol": symbol}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        ep = self.endpoint_for("dividends")
        return self.request(ep, params=params, endpoint_name="dividends")

    def get_splits(self, symbol: str, start: Optional[str], end: Optional[str]) -> Any:
        params: Dict[str, Any] = {"symbol": symbol}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        ep = self.endpoint_for("splits")
        return self.request(ep, params=params, endpoint_name="splits")

    def get_earnings(self, symbol: str) -> Any:
        ep = self.endpoint_for("earnings")
        return self.request(ep, params={"symbol": symbol}, endpoint_name="earnings")

    def get_earnings_calendar(self, start: str, end: str) -> Any:
        ep = self.endpoint_for("earnings_calendar")
        return self.request(ep, params={"from": start, "to": end}, endpoint_name="earnings_calendar")

    def get_earnings_surprises(self, symbol: str) -> Any:
        ep = self.endpoint_for("earnings_surprises")
        # Best-effort endpoint: avoid long retry tails when unavailable.
        return self.request(ep, params={"symbol": symbol}, endpoint_name="earnings_surprises", max_attempts=1)

    def get_income_statement(self, symbol: str, period: str = "quarter", limit: int = 40) -> Any:
        ep = self.endpoint_for("income_statement")
        return self.request(
            ep,
            params={"symbol": symbol, "period": period, "limit": limit},
            endpoint_name="income_statement",
        )

    def get_balance_sheet(self, symbol: str, period: str = "quarter", limit: int = 40) -> Any:
        ep = self.endpoint_for("balance_sheet")
        return self.request(
            ep,
            params={"symbol": symbol, "period": period, "limit": limit},
            endpoint_name="balance_sheet",
        )

    def get_cash_flow(self, symbol: str, period: str = "quarter", limit: int = 40) -> Any:
        ep = self.endpoint_for("cash_flow")
        return self.request(
            ep,
            params={"symbol": symbol, "period": period, "limit": limit},
            endpoint_name="cash_flow",
        )

    def get_stock_news(self, symbol: str, limit: int = 50) -> Any:
        ep = self.endpoint_for("stock_news")
        return self.request(
            ep,
            params={"symbol": symbol, "limit": limit},
            endpoint_name="stock_news",
        )

    def get_general_news(self, limit: int = 100) -> Any:
        ep = self.endpoint_for("general_news")
        return self.request(
            ep,
            params={"limit": limit},
            endpoint_name="general_news",
        )
