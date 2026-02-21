from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.migrate import pack_bundle, restore_bundle


def test_pack_and_restore_bundle(tmp_path: Path) -> None:
    src = tmp_path / "src_proj"
    dst = tmp_path / "dst_proj"
    run_id = "r1"

    (src / "artifacts" / "runs" / run_id).mkdir(parents=True, exist_ok=True)
    (src / "state").mkdir(parents=True, exist_ok=True)
    (src / "data" / "cache_http").mkdir(parents=True, exist_ok=True)
    (src / "conf").mkdir(parents=True, exist_ok=True)

    (src / "artifacts" / "runs" / run_id / "metrics.json").write_text(json.dumps({"ok": True}))
    (src / "state" / "last_run_state.json").write_text(json.dumps({"run_id": run_id}))
    (src / "data" / "cache_http" / "a.json").write_text("{}")
    (src / "conf" / "default.yaml").write_text("run: {}\n")
    (src / "conf" / "cost_model.yaml").write_text("commission_bps: 1.0\n")

    archive = tmp_path / "bundle.zip"
    packed = pack_bundle(project_dir=src, run_id=run_id, out_zip=archive)
    assert Path(packed["archive"]).exists()

    restored = restore_bundle(archive=Path(packed["archive"]), project_dir=dst)
    assert "artifacts/runs/r1/metrics.json" in restored["restored"]
    assert (dst / "artifacts" / "runs" / run_id / "metrics.json").exists()
