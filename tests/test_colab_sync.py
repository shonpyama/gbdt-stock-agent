from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.colab import restore_runtime_from_drive, sync_runtime_to_drive


def test_local_drive_sync_roundtrip(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    drive_root = tmp_path / "drive"

    (local_root / "data" / "cache_http").mkdir(parents=True, exist_ok=True)
    (local_root / "state").mkdir(parents=True, exist_ok=True)
    (local_root / "reports").mkdir(parents=True, exist_ok=True)
    (local_root / "data" / "cache_http" / "x.json").write_text("{}")
    (local_root / "state" / "last_run_state.json").write_text("{}")
    (local_root / "reports" / "r.md").write_text("# report")

    sync_stats = sync_runtime_to_drive(local_root=local_root, drive_path=drive_root)
    assert sync_stats["copied_files"] >= 2
    assert (drive_root / "data" / "cache_http" / "x.json").exists()

    # remove local runtime and restore from drive
    shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    restore_stats = restore_runtime_from_drive(local_root=local_root, drive_path=drive_root)
    assert restore_stats["copied_files"] >= 1
    assert (local_root / "data" / "cache_http" / "x.json").exists()
    assert (local_root / "data" / "cache_http").is_symlink()
    assert (local_root / "reports" / "r.md").exists()


def test_sync_copies_changed_metadata_even_if_size_and_mtime_match(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    drive_root = tmp_path / "drive"
    src = local_root / "state" / "last_run_state.json"
    dst = drive_root / "state" / "last_run_state.json"

    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text('{"stage":"a"}')
    sync_runtime_to_drive(local_root=local_root, drive_path=drive_root)
    assert dst.read_text() == src.read_text()

    src.write_text('{"stage":"b"}')  # same byte length
    st = dst.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))

    stats = sync_runtime_to_drive(local_root=local_root, drive_path=drive_root)
    assert stats["copied_files"] >= 1
    assert dst.read_text() == '{"stage":"b"}'


def test_quick_sync_rejects_unsafe_run_id(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    drive_root = tmp_path / "drive"
    local_root.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        sync_runtime_to_drive(local_root=local_root, drive_path=drive_root, mode="quick", run_id="../bad")
