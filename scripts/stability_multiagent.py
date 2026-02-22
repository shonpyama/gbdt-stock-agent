#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command_failed rc={proc.returncode}: {' '.join(shlex.quote(x) for x in cmd)}")


def _default_candidates(mode: str) -> List[str]:
    if mode == "model":
        return ["baseline_auto", "compact_31"]
    return ["baseline_current", "lb_1_5_20_60_120_shift2"]


def _script_for_mode(mode: str) -> str:
    return "scripts/model_stability_prod.py" if mode == "model" else "scripts/feature_stability_prod.py"


def _promote_script_for_mode(mode: str) -> str:
    return "scripts/promote_stability_model.py" if mode == "model" else "scripts/promote_stability_feature.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sharded stability evaluation in parallel using isolated git worktrees.")
    parser.add_argument("--mode", choices=["model", "feature"], required=True)
    parser.add_argument("--base-conf", default="conf/default.yaml")
    parser.add_argument("--end-dates", default="", help="Comma-separated end dates (sets *_STABILITY_END_DATES env)")
    parser.add_argument("--candidates", default="", help="Comma-separated candidate names")
    parser.add_argument("--worktree-root", default="/tmp/gbdt_stability_worktrees")
    parser.add_argument("--reports-prefix", default="", help="Prefix under reports/, default: <mode>_stability_multiagent")
    parser.add_argument("--promote-default", action="store_true", help="Promote selected merged result to default config")
    parser.add_argument("--keep-worktrees", action="store_true", help="Do not remove created worktrees")
    args = parser.parse_args()

    mode = args.mode
    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()] if args.candidates else _default_candidates(mode)
    if not candidates:
        raise ValueError("No candidates resolved")

    worktree_root = Path(args.worktree_root).resolve()
    worktree_root.mkdir(parents=True, exist_ok=True)
    reports_prefix = args.reports_prefix or f"{mode}_stability_multiagent"
    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    script_rel = _script_for_mode(mode)
    env_name = "MODEL_STABILITY_END_DATES" if mode == "model" else "FEATURE_STABILITY_END_DATES"
    shard_jsons: List[Path] = []
    shard_mds: List[Path] = []
    procs: List[subprocess.Popen] = []
    created_worktrees: List[Path] = []

    try:
        for c in candidates:
            wt = worktree_root / f"{mode}_{c}"
            if wt.exists():
                _run(["git", "worktree", "remove", "--force", str(wt)], cwd=PROJECT_DIR)
            _run(["git", "worktree", "add", "--detach", str(wt), "HEAD"], cwd=PROJECT_DIR)
            created_worktrees.append(wt)

            out_json = reports_dir / f"{reports_prefix}_{c}.json"
            out_md = reports_dir / f"{reports_prefix}_{c}.md"
            shard_jsons.append(out_json)
            shard_mds.append(out_md)

            cmd = [
                sys.executable,
                script_rel,
                "--base-conf",
                args.base_conf,
                "--only-candidates",
                c,
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
            env = os.environ.copy()
            if args.end_dates:
                env[env_name] = args.end_dates

            proc = subprocess.Popen(cmd, cwd=str(wt), env=env)
            procs.append(proc)

        rc = 0
        for p in procs:
            p.wait()
            if p.returncode != 0:
                rc = p.returncode
        if rc != 0:
            raise RuntimeError(f"One or more shard runs failed rc={rc}")

        merged_json = reports_dir / f"{reports_prefix}_merged.json"
        merged_md = reports_dir / f"{reports_prefix}_merged.md"
        _run(
            [
                sys.executable,
                "scripts/stability_merge.py",
                "--inputs",
                ",".join(str(p) for p in shard_jsons),
                "--out-json",
                str(merged_json),
                "--out-md",
                str(merged_md),
            ],
            cwd=PROJECT_DIR,
        )

        if args.promote_default:
            _run(
                [
                    sys.executable,
                    _promote_script_for_mode(mode),
                    "--results",
                    str(merged_json),
                    "--default-conf",
                    args.base_conf,
                ],
                cwd=PROJECT_DIR,
            )

        print(
            "\n".join(
                [
                    f"mode={mode}",
                    f"candidates={','.join(candidates)}",
                    f"shard_jsons={','.join(str(p) for p in shard_jsons)}",
                    f"merged_json={merged_json}",
                    f"merged_md={merged_md}",
                    f"promote_default={bool(args.promote_default)}",
                ]
            )
        )
        return 0
    finally:
        if not args.keep_worktrees:
            for wt in created_worktrees:
                subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=str(PROJECT_DIR), check=False)


if __name__ == "__main__":
    raise SystemExit(main())
