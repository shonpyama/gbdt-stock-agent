from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.models.gbdt import GBDTRegressor


def test_lightgbm_param_selection_prefers_gpu_when_available() -> None:
    p, gpu = GBDTRegressor._select_lightgbm_params({"n_estimators": 100}, prefer_gpu=True, gpu_available=True)
    assert p.get("device_type") == "gpu"
    assert gpu is True


def test_lightgbm_param_selection_respects_explicit_cpu_device() -> None:
    p, gpu = GBDTRegressor._select_lightgbm_params({"device_type": "cpu"}, prefer_gpu=True, gpu_available=True)
    assert p.get("device_type") == "cpu"
    assert gpu is False


def test_lightgbm_roundtrip_can_predict(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")

    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(200, 8))
    y_train = x_train[:, 0] * 0.4 - x_train[:, 1] * 0.2 + rng.normal(scale=0.05, size=200)
    x_val = rng.normal(size=(60, 8))
    y_val = x_val[:, 0] * 0.4 - x_val[:, 1] * 0.2 + rng.normal(scale=0.05, size=60)

    model = GBDTRegressor(
        seed=42,
        framework="lightgbm",
        params={"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31},
    )
    model.fit(x_train, y_train, x_val, y_val)
    assert model.train_accelerator in {"cpu", "gpu"}
    assert isinstance(model.trained_params, dict)

    path = tmp_path / "gbdt_model.txt"
    model.save(path)

    restored = GBDTRegressor.load(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = restored.predict(x_val[:10])
    assert pred.shape == (10,)
    assert np.isfinite(pred).all()
    assert not any("valid feature names" in str(w.message) for w in caught)
