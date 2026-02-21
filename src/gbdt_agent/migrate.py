"""Migration bundle utilities for local <-> colab transition."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pack_bundle(*, project_dir: Path, run_id: str, out_zip: Path) -> Dict[str, object]:
    project_dir = Path(project_dir)
    out_zip = Path(out_zip)

    include = [
        project_dir / "artifacts" / "runs" / run_id,
        project_dir / "state" / "last_run_state.json",
        project_dir / "data" / "cache_http",
        project_dir / "conf" / "default.yaml",
        project_dir / "conf" / "cost_model.yaml",
    ]

    included_rel: List[str] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "bundle"
        root.mkdir(parents=True, exist_ok=True)

        for p in include:
            if not p.exists():
                continue
            rel = p.relative_to(project_dir)
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if p.is_dir():
                shutil.copytree(p, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(p, dst)
            included_rel.append(str(rel))

        manifest = {
            "created_at": _utc_iso(),
            "run_id": run_id,
            "included": included_rel,
        }
        (root / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True))
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        archive = shutil.make_archive(str(out_zip.with_suffix("")), "zip", root_dir=root)

    return {
        "archive": archive,
        "run_id": run_id,
        "included": included_rel,
    }


def restore_bundle(*, archive: Path, project_dir: Path) -> Dict[str, object]:
    archive = Path(archive)
    project_dir = Path(project_dir)
    if not archive.exists():
        raise FileNotFoundError(str(archive))

    restored: List[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        shutil.unpack_archive(str(archive), str(tmp))
        for p in tmp.rglob("*"):
            if p.is_dir() or p.name == "bundle_manifest.json":
                continue
            rel = p.relative_to(tmp)
            dst = project_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            restored.append(str(rel))

        manifest = tmp / "bundle_manifest.json"
        if manifest.exists():
            m_dst = project_dir / "reports" / "bundle_manifest_restored.json"
            m_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, m_dst)

    return {
        "archive": str(archive),
        "project_dir": str(project_dir),
        "restored": restored,
    }
