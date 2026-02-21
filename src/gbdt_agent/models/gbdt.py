from __future__ import annotations

import json
import logging
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore


try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None  # type: ignore


@dataclass
class GBDTRegressor:
    seed: int = 42
    framework: str = "lightgbm"  # lightgbm | xgboost | sklearn
    params: Optional[Dict[str, Any]] = None
    prefer_gpu: bool = True
    model: Any = None
    trained_params: Optional[Dict[str, Any]] = None
    train_accelerator: str = "cpu"
    gpu_attempted: bool = False

    @staticmethod
    def _has_nvidia_gpu() -> bool:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-L"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3,
            )
            return "GPU " in out
        except Exception:
            return False

    @staticmethod
    def _is_gpu_device(value: Any) -> bool:
        if value is None:
            return False
        return str(value).strip().lower() in {"gpu", "cuda"}

    @staticmethod
    def _select_lightgbm_params(params: Dict[str, Any], *, prefer_gpu: bool, gpu_available: bool) -> tuple[Dict[str, Any], bool]:
        out = dict(params)
        if "device_type" in out or "device" in out:
            dev = out.get("device_type", out.get("device"))
            return out, GBDTRegressor._is_gpu_device(dev)
        if prefer_gpu and gpu_available:
            out["device_type"] = "gpu"
            return out, True
        return out, False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        params = dict(self.params or {})

        if self.framework == "lightgbm" and lgb is not None:
            default = {
                "n_estimators": 5000,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "random_state": self.seed,
                "n_jobs": -1,
            }
            requested_params = dict(default)
            requested_params.update(params)
            explicit_device = "device_type" in params or "device" in params

            fit_params, gpu_requested = self._select_lightgbm_params(
                requested_params,
                prefer_gpu=bool(self.prefer_gpu),
                gpu_available=self._has_nvidia_gpu(),
            )
            self.gpu_attempted = gpu_requested

            def _fit_with(p: Dict[str, Any]) -> None:
                self.model = lgb.LGBMRegressor(**p)
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                )

            if gpu_requested and not explicit_device:
                try:
                    _fit_with(fit_params)
                    self.trained_params = dict(fit_params)
                    self.train_accelerator = "gpu"
                    return
                except Exception as exc:
                    logger.warning(
                        "lightgbm_gpu_auto_fallback_to_cpu reason=%s",
                        type(exc).__name__,
                    )
                    cpu_params = dict(requested_params)
                    cpu_params.pop("device_type", None)
                    cpu_params.pop("device", None)
                    _fit_with(cpu_params)
                    self.trained_params = dict(cpu_params)
                    self.train_accelerator = "cpu"
                    return

            _fit_with(fit_params)
            self.trained_params = dict(fit_params)
            dev = fit_params.get("device_type", fit_params.get("device"))
            self.train_accelerator = "gpu" if self._is_gpu_device(dev) else "cpu"
            return

        if self.framework == "xgboost" and xgb is not None:
            default = {
                "n_estimators": 5000,
                "learning_rate": 0.03,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "random_state": self.seed,
                "n_jobs": -1,
                "tree_method": "hist",
            }
            default.update(params)
            self.model = xgb.XGBRegressor(**default)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=50,
            )
            self.trained_params = dict(default)
            device = default.get("device")
            tree_method = str(default.get("tree_method", "")).strip().lower()
            self.train_accelerator = "gpu" if self._is_gpu_device(device) or tree_method == "gpu_hist" else "cpu"
            self.gpu_attempted = self.train_accelerator == "gpu"
            return

        if GradientBoostingRegressor is None:
            self.framework = "linear"
            x_aug = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)
            coef, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
            self.model = {"coef": coef.tolist()}
            self.trained_params = {}
            self.train_accelerator = "cpu"
            self.gpu_attempted = False
            return
        self.framework = "sklearn"
        default = {
            "n_estimators": int(params.get("n_estimators", 300)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "max_depth": int(params.get("max_depth", 3)),
            "random_state": self.seed,
        }
        self.model = GradientBoostingRegressor(**default)
        self.model.fit(X_train, y_train)
        self.trained_params = dict(default)
        self.train_accelerator = "cpu"
        self.gpu_attempted = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit")
        if self.framework == "linear":
            coef = np.asarray(self.model["coef"], dtype=float)
            x_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
            return np.asarray(x_aug @ coef, dtype=float)
        if self.framework == "lightgbm" and lgb is not None:
            x_arr = np.asarray(X, dtype=float)
            try:
                booster = getattr(self.model, "booster_", None) or getattr(self.model, "_Booster", None)
                if booster is not None:
                    feature_names = list(booster.feature_name() or [])
                    if feature_names and len(feature_names) == x_arr.shape[1]:
                        x_df = pd.DataFrame(x_arr, columns=feature_names)
                        return np.asarray(self.model.predict(x_df), dtype=float)
            except Exception:
                pass
            return np.asarray(self.model.predict(x_arr), dtype=float)
        return np.asarray(self.model.predict(X), dtype=float)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "framework": self.framework,
            "seed": self.seed,
            "params": self.params or {},
            "prefer_gpu": bool(self.prefer_gpu),
            "trained_params": self.trained_params or {},
            "train_accelerator": self.train_accelerator,
            "gpu_attempted": bool(self.gpu_attempted),
        }
        (path.with_suffix(path.suffix + ".meta.json")).write_text(json.dumps(meta, indent=2, ensure_ascii=True))

        if self.framework == "lightgbm":
            booster = self.model.booster_
            booster.save_model(str(path))
            return
        if self.framework == "xgboost":
            self.model.save_model(str(path))
            return
        with path.open("wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str | Path) -> "GBDTRegressor":
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        framework = meta.get("framework", "lightgbm")
        seed = int(meta.get("seed", 42))
        params = meta.get("params", {})
        obj = cls(seed=seed, framework=framework, params=params, prefer_gpu=bool(meta.get("prefer_gpu", True)))
        if isinstance(meta.get("trained_params"), dict):
            obj.trained_params = meta.get("trained_params")
        obj.train_accelerator = str(meta.get("train_accelerator", "cpu"))
        obj.gpu_attempted = bool(meta.get("gpu_attempted", False))

        if framework == "lightgbm":
            if lgb is None:
                raise RuntimeError("LightGBM not available for load")
            booster = lgb.Booster(model_file=str(path))
            wrapper = lgb.LGBMRegressor(**(params or {}))
            wrapper._Booster = booster
            # Mark as fitted so sklearn compatibility checks pass on predict().
            wrapper.fitted_ = True
            n_features = booster.num_feature()
            wrapper._n_features = n_features
            wrapper._n_features_in = n_features
            obj.model = wrapper
            return obj

        if framework == "xgboost":
            if xgb is None:
                raise RuntimeError("XGBoost not available for load")
            model = xgb.XGBRegressor(**(params or {}))
            model.load_model(str(path))
            obj.model = model
            return obj

        with path.open("rb") as f:
            obj.model = pickle.load(f)
        return obj
