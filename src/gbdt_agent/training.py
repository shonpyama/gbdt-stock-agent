from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .models.gbdt import GBDTRegressor


logger = logging.getLogger(__name__)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _build_xy(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    X = df[list(feature_cols)].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[m], y[m]


@dataclass(frozen=True)
class TrainResult:
    model_ckpt_paths: Dict[str, str]
    training_info: Dict[str, Any]


def train_models(
    *,
    cfg: Dict[str, Any],
    run_dir: Path,
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    train_dates: Sequence,
    val_dates: Sequence,
) -> TrainResult:
    seed = int(cfg.get("run", {}).get("seed", 42))
    set_global_seeds(seed)
    horizon = int((cfg.get("labels") or {}).get("horizon_trading_days", 20))
    target_col = str((cfg.get("labels") or {}).get("target_col", f"future_return_{horizon}d"))

    models_cfg = cfg.get("models", {}) or {}
    out_dir = run_dir / "model_ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = merged[merged["decision_date"].isin(train_dates)]
    val_df = merged[merged["decision_date"].isin(val_dates)]
    X_train, y_train = _build_xy(train_df, feature_cols, target_col)
    X_val, y_val = _build_xy(val_df, feature_cols, target_col)
    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError(f"Insufficient train/val rows after filtering finite values train={len(X_train)} val={len(X_val)}")

    if len(X_train) < 1000 or len(X_val) < 200:
        logger.warning(f"small_train_or_val_samples train={len(X_train)} val={len(X_val)}")

    ckpts: Dict[str, str] = {}
    info: Dict[str, Any] = {
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_features": int(len(feature_cols)),
        "feature_cols": list(feature_cols),
        "target_col": target_col,
    }
    if not bool(models_cfg.get("gbdt", {}).get("enabled", True)):
        raise RuntimeError("models.gbdt.enabled must be true for this agent")

    mcfg = models_cfg.get("gbdt", {}) or {}
    framework = str(mcfg.get("framework", "lightgbm"))
    params = mcfg.get("params", {}) or {}
    prefer_gpu = bool(mcfg.get("prefer_gpu", True))
    model = GBDTRegressor(seed=seed, framework=framework, params=params, prefer_gpu=prefer_gpu)
    model.fit(X_train, y_train, X_val, y_val)
    yhat_val = model.predict(X_val)
    val_mse = float(np.mean((yhat_val - y_val) ** 2))
    if model.framework == "lightgbm":
        path = out_dir / "gbdt_model.txt"
    elif model.framework == "xgboost":
        path = out_dir / "gbdt_model.json"
    else:
        path = out_dir / "gbdt_model.pkl"
    model.save(path)
    ckpts["gbdt"] = str(path)
    info["gbdt"] = {
        "framework": model.framework,
        "prefer_gpu": prefer_gpu,
        "gpu_attempted": bool(model.gpu_attempted),
        "accelerator": model.train_accelerator,
        "params": params,
        "effective_params": model.trained_params or params,
        "val_mse": val_mse,
    }

    return TrainResult(model_ckpt_paths=ckpts, training_info=info)
