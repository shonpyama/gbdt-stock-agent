"""Google Colab local-first persistence helpers."""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import os
import re
import shutil
import threading
import time
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
SYNC_DIRS_QUICK = (
    "state",
    "reports",
    "logs",
)
CONTENT_CHECK_SUFFIXES = {".json", ".md", ".txt", ".log", ".yaml", ".yml"}
CONTENT_CHECK_BASENAMES = {"last_run_state.json", "lock.json", "metrics.json", "report.md", "artifact_manifest.json"}
CONTENT_CHECK_SKIP_PARTS = {"cache_http"}
SYNC_LOCK_FILE = ".colab_sync.lock"
SYNC_LOCK_STALE_SECONDS = 4 * 60 * 60
SYNC_LOCK_WAIT_SECONDS = 60.0
SYNC_LOCK_POLL_SECONDS = 0.5


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

    src_stat = src.stat()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return 1

    dst_stat = dst.stat()
    src_mtime_ns = getattr(src_stat, "st_mtime_ns", int(src_stat.st_mtime * 1_000_000_000))
    dst_mtime_ns = getattr(dst_stat, "st_mtime_ns", int(dst_stat.st_mtime * 1_000_000_000))

    if src_stat.st_size != dst_stat.st_size:
        shutil.copy2(src, dst)
        return 1
    if src_mtime_ns > dst_mtime_ns:
        shutil.copy2(src, dst)
        return 1
    if src_mtime_ns < dst_mtime_ns:
        return 0

    # mtime/size が一致しても内容が変わるケースに備え、メタ情報系ファイルはハッシュで比較する。
    # cache_http はファイル数が多く、ハッシュ比較コストが高いためメタ情報チェック対象から除外する。
    should_hash_check = (
        src.name in CONTENT_CHECK_BASENAMES
        or src.suffix.lower() in CONTENT_CHECK_SUFFIXES
        or "state" in src.parts
    )
    if should_hash_check and CONTENT_CHECK_SKIP_PARTS.isdisjoint(set(src.parts)):
        if _sha1_file(src) != _sha1_file(dst):
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


def _read_last_run_id(root: Path) -> Optional[str]:
    p = root / "state" / "last_run_state.json"
    if not p.exists():
        return None
    try:
        import json

        payload = json.loads(p.read_text())
        rid = str(payload.get("run_id") or "").strip()
        if re.match(r"^[0-9]{8}_[0-9]{6}Z_[0-9a-f]{8,}_[0-9a-f]{8,}$", rid):
            return rid
    except Exception:
        return None
    return None


def _resolve_sync_dirs(*, mode: str, run_id: Optional[str], source_root: Path) -> tuple[str, ...]:
    m = str(mode or "full").strip().lower()
    if m not in {"full", "quick"}:
        m = "full"
    if m == "full":
        return SYNC_DIRS
    rid = ""
    run_id_raw = str(run_id or "").strip()
    if run_id_raw:
        rid = _sanitize_run_id(run_id_raw)
    else:
        rid = _read_last_run_id(source_root) or ""
    out = list(SYNC_DIRS_QUICK)
    if rid:
        out.append(f"artifacts/runs/{rid}")
    return tuple(out)


def _sanitize_run_id(run_id: str) -> str:
    rid = str(run_id).strip()
    if not rid:
        return ""
    if ".." in rid or "/" in rid or "\\" in rid:
        raise ValueError(f"Invalid run_id: {run_id}")
    if not re.match(r"^[A-Za-z0-9._-]+$", rid):
        raise ValueError(f"Invalid run_id: {run_id}")
    return rid


def _ensure_drive_cache_link(*, local_root: Path, drive_path: Path) -> Dict[str, object]:
    local_cache = local_root / "data" / "cache_http"
    drive_cache = drive_path / "data" / "cache_http"
    drive_cache.mkdir(parents=True, exist_ok=True)
    local_cache.parent.mkdir(parents=True, exist_ok=True)

    if local_cache.is_symlink():
        try:
            if local_cache.resolve() == drive_cache.resolve():
                return {"cache_linked": True, "cache_link_reason": "already_linked"}
        except Exception:
            pass

    backup_path: Optional[Path] = None
    try:
        if local_cache.exists() and local_cache.is_dir():
            # One-time seed into Drive cache.
            _sync_tree(local_cache, drive_cache)
            backup_path = local_cache.parent / f".cache_http_backup_{os.getpid()}_{int(time.time())}"
            local_cache.rename(backup_path)
        elif local_cache.exists():
            backup_path = local_cache.parent / f".cache_http_backup_{os.getpid()}_{int(time.time())}"
            local_cache.rename(backup_path)

        os.symlink(str(drive_cache), str(local_cache), target_is_directory=True)

        if backup_path and backup_path.exists():
            if backup_path.is_dir():
                shutil.rmtree(backup_path)
            else:
                backup_path.unlink()
        return {"cache_linked": True, "cache_link_reason": "linked_to_drive"}
    except Exception as exc:
        # Roll back to local cache if link creation failed.
        try:
            if local_cache.is_symlink() or local_cache.exists():
                if local_cache.is_dir() and not local_cache.is_symlink():
                    shutil.rmtree(local_cache)
                else:
                    local_cache.unlink()
            if backup_path and backup_path.exists():
                backup_path.rename(local_cache)
        except Exception:
            pass
        return {"cache_linked": False, "cache_link_reason": f"{type(exc).__name__}: {exc}"}


class _SyncLock:
    def __init__(self, drive_path: Path):
        self.lock_path = drive_path / SYNC_LOCK_FILE
        self.acquired = False

    def _prune_stale(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                age = max(0.0, time.time() - self.lock_path.stat().st_mtime)
                if age > float(SYNC_LOCK_STALE_SECONDS):
                    self.lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    def acquire(self, *, wait_seconds: float = SYNC_LOCK_WAIT_SECONDS) -> None:
        deadline = time.time() + max(0.0, float(wait_seconds))
        while True:
            self._prune_stale()
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, f"{os.getpid()} {time.time()}".encode("utf-8"))
                finally:
                    os.close(fd)
                self.acquired = True
                return
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"sync lock timeout: {self.lock_path}")
                time.sleep(float(SYNC_LOCK_POLL_SECONDS))

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        self.acquired = False


def restore_runtime_from_drive(
    local_root: Optional[Path] = None,
    drive_path: Optional[Path] = None,
    *,
    mode: str = "full",
    run_id: Optional[str] = None,
) -> Dict[str, object]:
    local_root = local_root or Path.cwd()
    drive_path = drive_path or DEFAULT_DRIVE_PATH
    local_root.mkdir(parents=True, exist_ok=True)
    drive_path.mkdir(parents=True, exist_ok=True)

    lock = _SyncLock(drive_path)
    lock.acquire()
    cache_info = _ensure_drive_cache_link(local_root=local_root, drive_path=drive_path)
    copied = 0
    try:
        rel_dirs = _resolve_sync_dirs(mode=mode, run_id=run_id, source_root=drive_path)
        for rel in rel_dirs:
            if str(rel) == "data/cache_http":
                continue
            copied += _sync_tree(drive_path / rel, local_root / rel)
        out = {
            "direction": "drive_to_local",
            "mode": str(mode),
            "run_id": str(run_id or ""),
            "copied_files": copied,
            "local_root": str(local_root),
            "drive_path": str(drive_path),
            "sync_dirs": list(rel_dirs),
        }
        out.update(cache_info)
        return out
    finally:
        lock.release()


def sync_runtime_to_drive(
    local_root: Optional[Path] = None,
    drive_path: Optional[Path] = None,
    *,
    mode: str = "full",
    run_id: Optional[str] = None,
) -> Dict[str, object]:
    local_root = local_root or Path.cwd()
    drive_path = drive_path or DEFAULT_DRIVE_PATH
    local_root.mkdir(parents=True, exist_ok=True)
    drive_path.mkdir(parents=True, exist_ok=True)

    lock = _SyncLock(drive_path)
    lock.acquire()
    cache_info = _ensure_drive_cache_link(local_root=local_root, drive_path=drive_path)
    copied = 0
    try:
        rel_dirs = _resolve_sync_dirs(mode=mode, run_id=run_id, source_root=local_root)
        for rel in rel_dirs:
            if str(rel) == "data/cache_http":
                continue
            copied += _sync_tree(local_root / rel, drive_path / rel)
        out = {
            "direction": "local_to_drive",
            "mode": str(mode),
            "run_id": str(run_id or ""),
            "copied_files": copied,
            "local_root": str(local_root),
            "drive_path": str(drive_path),
            "sync_dirs": list(rel_dirs),
        }
        out.update(cache_info)
        return out
    finally:
        lock.release()


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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
    os.environ["GBDT_CHECKPOINT_DRIVE_PATH"] = str(dp)
    restore_runtime_from_drive(drive_path=dp)
    handle = start_periodic_drive_sync(interval_seconds=int(interval_seconds), drive_path=dp)

    def _final_sync() -> None:
        try:
            handle.stop()
        except Exception:
            pass

    atexit.register(_final_sync)
    return handle
