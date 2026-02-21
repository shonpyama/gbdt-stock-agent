from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


STAGES_SUCCESS_ORDER = [
    "stage_00_env_ready",
    "stage_10_data_ready",
    "stage_20_validation_passed",
    "stage_30_features_ready",
    "stage_40_split_leakcheck_passed",
    "stage_50_models_trained",
    "stage_60_predictions_ready",
    "stage_70_backtest_ready",
    "stage_80_report_ready",
]

FAILED_TO_PREV_SUCCESS = {
    "stage_20_validation_failed": "stage_10_data_ready",
    "stage_40_split_leakcheck_failed": "stage_30_features_ready",
}

STATE_DEFAULTS: Dict[str, Any] = {
    "run_id": None,
    "stage": "stage_00_env_ready",
    "conf_hash": None,
    "dataset_id": None,
    "feature_store_id": None,
    "train_window": None,
    "val_window": None,
    "test_window": None,
    "model_ckpt_paths": {},
    "last_data_date": None,
    "errors": [],
    "code_hash": None,
    "env_snapshot_path": None,
    "artifacts_dir": None,
    "updated_at": None,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(STATE_DEFAULTS)
    state.update(payload or {})

    if state.get("stage") not in STAGES_SUCCESS_ORDER and state.get("stage") not in FAILED_TO_PREV_SUCCESS:
        state["stage"] = "stage_00_env_ready"
    if not isinstance(state.get("model_ckpt_paths"), dict):
        state["model_ckpt_paths"] = {}
    if not isinstance(state.get("errors"), list):
        state["errors"] = []
    return state


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    tmp.replace(path)


def stage_success_index(stage: str) -> int:
    return STAGES_SUCCESS_ORDER.index(stage)


def last_success_stage(stage: str) -> Optional[str]:
    if stage in STAGES_SUCCESS_ORDER:
        return stage
    if stage in FAILED_TO_PREV_SUCCESS:
        return FAILED_TO_PREV_SUCCESS[stage]
    return None


def is_success_stage_done(state: Dict[str, Any], stage: str) -> bool:
    st = state.get("stage")
    last = last_success_stage(str(st)) if st else None
    if not last:
        return False
    return stage_success_index(last) >= stage_success_index(stage)


def last_run_state_path(state_dir: Path) -> Path:
    return state_dir / "last_run_state.json"


def load_last_state(state_dir: Path) -> Optional[Dict[str, Any]]:
    path = last_run_state_path(state_dir)
    if not path.exists():
        return None
    loaded = json.loads(path.read_text())
    return normalize_state(loaded)


def save_last_state(state_dir: Path, payload: Dict[str, Any]) -> Path:
    path = last_run_state_path(state_dir)
    state = normalize_state(payload)
    state["updated_at"] = utc_now_iso()
    _atomic_write_json(path, state)
    return path


def append_state_error(payload: Dict[str, Any], *, stage: str, error: Any) -> Dict[str, Any]:
    state = normalize_state(payload)
    errs = list(state.get("errors") or [])
    errs.append({
        "stage": stage,
        "error": error,
        "occurred_at": utc_now_iso(),
    })
    state["errors"] = errs[-50:]
    return state


@dataclass(frozen=True)
class LockInfo:
    run_id: str
    created_at: str
    host: str
    pid: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "host": self.host,
            "pid": self.pid,
        }


def lock_path(state_dir: Path) -> Path:
    return state_dir / "lock.json"


def acquire_lock(state_dir: Path, run_id: str, force: bool = False) -> LockInfo:
    state_dir.mkdir(parents=True, exist_ok=True)
    lp = lock_path(state_dir)
    if lp.exists() and not force:
        existing = json.loads(lp.read_text())
        raise RuntimeError(f"Lock exists: {existing}")
    if lp.exists() and force:
        lp.unlink()

    info = LockInfo(
        run_id=run_id,
        created_at=utc_now_iso(),
        host=socket.gethostname(),
        pid=str(os.getpid()),
    )
    with lp.open("x") as f:
        f.write(json.dumps(info.to_dict(), indent=2, ensure_ascii=True))
    return info


def release_lock(state_dir: Path) -> None:
    lp = lock_path(state_dir)
    try:
        lp.unlink()
    except FileNotFoundError:
        return
