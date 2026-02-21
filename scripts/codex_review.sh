#!/usr/bin/env bash
set -euo pipefail

if ! command -v codex >/dev/null 2>&1; then
  echo "Error: codex command not found in PATH." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/codex_review.sh "task prompt"

Optional env vars:
  CODEX_REVIEW_FLAGS  Extra flags for codex exec.
EOF
  exit 1
fi

TASK_PROMPT="$1"
DEFAULT_FLAGS="--skip-git-repo-check --dangerously-bypass-approvals-and-sandbox"
FLAGS="${CODEX_REVIEW_FLAGS:-$DEFAULT_FLAGS}"

PROMPT=$(cat <<EOF
あなたは厳密なコードレビュアーです。以下を優先してください。
1. バグ・仕様逸脱・回帰リスク
2. テスト不足
3. 例外系/境界値の欠落

対象タスク:
${TASK_PROMPT}

制約:
- 既存の設計方針を尊重
- 変更は最小限
- 変更後に実行した確認コマンドと結果を要約
EOF
)

echo "Running: codex exec ${FLAGS} <prompt>"
codex exec ${FLAGS} "${PROMPT}"
