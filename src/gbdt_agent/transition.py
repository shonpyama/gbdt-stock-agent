"""Pre-colab mandatory transition report."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _cmd(repo: Path, *args: str) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"(unavailable: {type(exc).__name__})"


def generate_transition_report(*, project_dir: Path, run_id: str, target: str) -> Path:
    project_dir = Path(project_dir)
    run_dir = project_dir / "artifacts" / "runs" / run_id
    metrics = _read_json(run_dir / "metrics.json")
    manifest = _read_json(run_dir / "artifact_manifest.json")
    state = _read_json(project_dir / "state" / "last_run_state.json")

    report_dir = project_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"pre_colab_transition_{_utc_ts()}.md"

    stage = str((state or {}).get("stage", "unknown"))
    status = str((metrics or {}).get("status", "unknown"))
    errors = (metrics or {}).get("errors") or []
    backtest = ((metrics or {}).get("backtest") or {}).get("summary") or {}
    leakage = (metrics or {}).get("leakage") or {}

    git_status = _cmd(project_dir, "status", "--short")
    git_commits = _cmd(project_dir, "log", "--oneline", "-n", "20")

    lines = [
        "# Pre-Colab Transition Report",
        "",
        f"- target: `{target}`",
        f"- run_id: `{run_id}`",
        f"- status: `{status}`",
        f"- current_stage: `{stage}`",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Local Validation Summary",
        "",
        f"- leakage_passed: `{leakage.get('passed')}`",
        f"- chosen_model: `{((metrics or {}).get('backtest') or {}).get('chosen_model')}`",
        f"- sharpe: `{backtest.get('sharpe')}`",
        f"- max_drawdown: `{backtest.get('max_drawdown')}`",
        f"- total_return: `{backtest.get('total_return')}`",
        "",
        "## Errors",
        "",
        f"- error_count: `{len(errors)}`",
    ]
    if errors:
        lines += ["```json", json.dumps(errors, indent=2, ensure_ascii=True), "```", ""]

    lines += [
        "## Git Diff Summary",
        "",
        "```",
        git_status or "(clean)",
        "```",
        "",
        "## Recent Commits",
        "",
        "```",
        git_commits or "(none)",
        "```",
        "",
        "## Artifact Manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2, ensure_ascii=True) if manifest else "{}",
        "```",
        "",
        "## Migration Risks",
        "",
        "- Colabランタイム再起動時に未同期データが失われる可能性。",
        "- FMP APIレート制限到達時にデータ更新ステージが遅延する可能性。",
        "- GPUランタイム差異で学習再現性が揺らぐ可能性。",
        "",
        "## Rollback Plan",
        "",
        "1. `python -m gbdt_agent.cli migrate pack --run-id <run_id> --out <zip>` でローカル状態を固定。",
        "2. Colabで失敗時は `python -m gbdt_agent.cli colab restore --drive-path <path>` で復元。",
        "3. `python -m gbdt_agent.cli run --conf conf/default.yaml --resume --force-stage <stage>` で再開。",
        "",
        "## Approval Gate",
        "",
        "Colab実行は**ユーザー明示承認後のみ**開始可能です。",
    ]

    out.write_text("\n".join(lines) + "\n")
    return out
