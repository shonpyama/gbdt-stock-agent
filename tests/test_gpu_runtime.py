from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_gpu_runtime_env_off(monkeypatch) -> None:
    monkeypatch.setenv("GBDT_ENABLE_CUDF_PANDAS", "0")
    mod = importlib.import_module("gbdt_agent.gpu_runtime")
    mod = importlib.reload(mod)

    state = mod.maybe_enable_cudf_pandas()
    assert state["enabled"] is False
    assert state["backend"] == "pandas"
    assert state["reason"] == "env_off"


def test_gpu_runtime_auto_attempt_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("GBDT_ENABLE_CUDF_PANDAS", raising=False)
    mod = importlib.import_module("gbdt_agent.gpu_runtime")
    mod = importlib.reload(mod)

    def _raise_import_error(name: str):  # type: ignore[no-untyped-def]
        raise ImportError("no cudf")

    monkeypatch.setattr(mod.importlib, "import_module", _raise_import_error)
    state = mod.maybe_enable_cudf_pandas()
    assert state["enabled"] is False
    assert state["backend"] == "pandas"
    assert str(state["reason"]).startswith("ImportError:")


def test_gpu_runtime_import_error_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("GBDT_ENABLE_CUDF_PANDAS", "1")
    mod = importlib.import_module("gbdt_agent.gpu_runtime")
    mod = importlib.reload(mod)

    def _raise_import_error(name: str):  # type: ignore[no-untyped-def]
        raise ImportError("no cudf")

    monkeypatch.setattr(mod.importlib, "import_module", _raise_import_error)
    state = mod.maybe_enable_cudf_pandas()
    assert state["enabled"] is False
    assert state["backend"] == "pandas"
    assert str(state["reason"]).startswith("ImportError:")
