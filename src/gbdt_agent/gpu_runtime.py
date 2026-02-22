from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict


logger = logging.getLogger(__name__)
_STATE: Dict[str, Any] = {
    "enabled": False,
    "backend": "pandas",
    "reason": "not_initialized",
}
_INITIALIZED = False
_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}


def _env_enabled(name: str) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    if not v:
        # Default to auto-enable when available unless explicitly disabled.
        return True
    if v in _FALSEY:
        return False
    return v in _TRUTHY


def maybe_enable_cudf_pandas() -> Dict[str, Any]:
    global _INITIALIZED
    if _INITIALIZED:
        return dict(_STATE)
    _INITIALIZED = True

    if not _env_enabled("GBDT_ENABLE_CUDF_PANDAS"):
        _STATE.update({"enabled": False, "backend": "pandas", "reason": "env_off"})
        return dict(_STATE)

    try:
        mod = importlib.import_module("cudf.pandas")
        install = getattr(mod, "install", None)
        if not callable(install):
            raise RuntimeError("cudf.pandas.install not callable")
        install()
        _STATE.update({"enabled": True, "backend": "cudf.pandas", "reason": "ok"})
        logger.info("cudf_pandas_enabled")
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _STATE.update({"enabled": False, "backend": "pandas", "reason": reason[:240]})
        logger.warning("cudf_pandas_enable_failed reason=%s", reason)

    return dict(_STATE)


def dataframe_backend_state() -> Dict[str, Any]:
    return dict(_STATE)
