from __future__ import annotations

import sys
from pathlib import Path

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
    assert sync_stats["copied_files"] >= 3

    # remove local and restore from drive
    (local_root / "data" / "cache_http" / "x.json").unlink()
    (local_root / "reports" / "r.md").unlink()
    restore_stats = restore_runtime_from_drive(local_root=local_root, drive_path=drive_root)
    assert restore_stats["copied_files"] >= 2
    assert (local_root / "data" / "cache_http" / "x.json").exists()
    assert (local_root / "reports" / "r.md").exists()
