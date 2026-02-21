from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import yaml


SECRET_KEYS = {"api_key", "apikey", "fmp_api_key", "key", "token", "secret"}


def _normalize_config(obj: Any, secret_keys: Iterable[str] = SECRET_KEYS) -> Any:
    if isinstance(obj, Mapping):
        clean: Dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in secret_keys:
                continue
            clean[str(k)] = _normalize_config(v, secret_keys)
        return clean
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_normalize_config(v, secret_keys) for v in obj]
    return obj


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(cfg: Mapping[str, Any]) -> str:
    clean = _normalize_config(cfg)
    payload = stable_json_dumps(clean)
    return sha1_hex(payload)


def symbols_hash(symbols: Sequence[str]) -> str:
    return sha1_hex(",".join(sorted(symbols)))[:12]


def deep_merge(base: MutableMapping[str, Any], upd: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in upd.items():
        if (
            k in base
            and isinstance(base[k], MutableMapping)
            and isinstance(v, Mapping)
        ):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return raw


def load_config(conf_path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_yaml(Path(conf_path))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
