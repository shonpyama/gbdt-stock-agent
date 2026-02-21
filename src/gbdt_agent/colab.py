"""Google Colab local-first persistence helpers."""

from __future__ import annotations

import atexit
import importlib.util
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

DEFAULT_DRIVE_PATH = Path("/content/drive/MyDrive/gbdt-stock-agent")
SYNC_DIRS = (
    "data/raw",
    "data/processed",
    "data/feature_store",
    "data/cache_http",
    "artifacts/runs",
    "state",
    "logs",
    "reports",
)


def is_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def mount_drive(drive_path: Optional[Path] = None) -> Path:
    out = drive_path or DEFAULT_DRIVE_PATH
    if is_colab():
        from google.colab import drive

        drive.mount("/content/drive")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _copy_if_newer(src: Path, dst: Path) -> int:
    if not src.exists() or not src.is_file():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or int(src.stat().st_mtime) > int(dst.stat().st_mtime) or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)
        return 1
    return 0


def _sync_tree(src_root: Path, dst_root: Path) -> int:
    if not src_root.exists():
        return 0
    copied = 0
    for p in src_root.rglob("*"):
        if p.is_file():
            copied += _copy_if_newer(p, dst_root / p.relative_to(src_root))
    return copied


def restore_runtime_from_drive(local_root: Optional[Path] = None, drive_path: Optional[Path] = None) -> Dict[str, object]:
    local_root = local_root or Path.cwd()
    drive_path = drive_path or DEFAULT_DRIVE_PATH
    local_root.mkdir(parents=True, exist_ok=True)
    drive_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    for rel in SYNC_DIRS:
        copied += _sync_tree(drive_path / rel, local_root / rel)
    return {
        "direction": "drive_to_local",
        "copied_files": copied,
        "local_root": str(local_root),
        "drive_path": str(drive_path),
    }


def sync_runtime_to_drive(local_root: Optional[Path] = None, drive_path: Optional[Path] = None) -> Dict[str, object]:
    local_root = local_root or Path.cwd()
    drive_path = drive_path or DEFAULT_DRIVE_PATH
    local_root.mkdir(parents=True, exist_ok=True)
    drive_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    for rel in SYNC_DIRS:
        copied += _sync_tree(local_root / rel, drive_path / rel)
    return {
        "direction": "local_to_drive",
        "copied_files": copied,
        "local_root": str(local_root),
        "drive_path": str(drive_path),
    }


@dataclass
class PeriodicSyncHandle:
    local_root: Path
    drive_path: Path
    interval_seconds: int
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    last_sync: Optional[Dict[str, object]] = None
    last_error: Optional[str] = None

    def sync_now(self) -> Dict[str, object]:
        with self._lock:
            self.last_sync = sync_runtime_to_drive(local_root=self.local_root, drive_path=self.drive_path)
            self.last_error = None
            return self.last_sync

    def stop(self) -> Dict[str, object]:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(1, min(self.interval_seconds, 5)))
        return self.sync_now()


def start_periodic_drive_sync(
    *,
    interval_seconds: int = 600,
    local_root: Optional[Path] = None,
    drive_path: Optional[Path] = None,
) -> PeriodicSyncHandle:
    local_root = local_root or Path.cwd()
    drive_path = drive_path or DEFAULT_DRIVE_PATH

    handle = PeriodicSyncHandle(local_root=local_root, drive_path=drive_path, interval_seconds=int(interval_seconds))

    def _worker() -> None:
        while not handle._stop_event.wait(handle.interval_seconds):
            try:
                handle.sync_now()
            except Exception as exc:  # pragma: no cover
                handle.last_error = str(exc)

    handle._thread = threading.Thread(target=_worker, name="gbdt-drive-sync", daemon=True)
    handle._thread.start()
    return handle


def setup_fast_colab_persistence(
    *,
    drive_path: Optional[Path] = None,
    interval_seconds: int = 600,
) -> PeriodicSyncHandle:
    dp = mount_drive(drive_path)
    restore_runtime_from_drive(drive_path=dp)
    handle = start_periodic_drive_sync(interval_seconds=int(interval_seconds), drive_path=dp)

    def _final_sync() -> None:
        try:
            handle.stop()
        except Exception:
            pass

    atexit.register(_final_sync)
    return handle
