from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


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
    model: Any = None

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
            default.update(params)
            self.model = lgb.LGBMRegressor(**default)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
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
            return

        if GradientBoostingRegressor is None:
            self.framework = "linear"
            x_aug = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)
            coef, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
            self.model = {"coef": coef.tolist()}
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit")
        if self.framework == "linear":
            coef = np.asarray(self.model["coef"], dtype=float)
            x_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
            return np.asarray(x_aug @ coef, dtype=float)
        return np.asarray(self.model.predict(X), dtype=float)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {"framework": self.framework, "seed": self.seed, "params": self.params or {}}
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
        obj = cls(seed=seed, framework=framework, params=params)

        if framework == "lightgbm":
            if lgb is None:
                raise RuntimeError("LightGBM not available for load")
            booster = lgb.Booster(model_file=str(path))
            wrapper = lgb.LGBMRegressor(**(params or {}))
            wrapper._Booster = booster
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
