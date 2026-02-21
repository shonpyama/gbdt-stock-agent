#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LABEL="${1:-manual}"
TS="$(date -u +%Y%m%d_%H%M%S)"
SAFE_LABEL="$(printf "%s" "$LABEL" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
OUT_DIR="outputs/checkpoints/${TS}_${SAFE_LABEL}"
mkdir -p "$OUT_DIR"

echo "Saving checkpoint to: $OUT_DIR"

git rev-parse --abbrev-ref HEAD > "$OUT_DIR/git_branch.txt" 2>/dev/null || true
git rev-parse HEAD > "$OUT_DIR/git_head.txt" 2>/dev/null || true
git status --short > "$OUT_DIR/git_status_short.txt" 2>/dev/null || true
git status > "$OUT_DIR/git_status.txt" 2>/dev/null || true
git diff > "$OUT_DIR/git_diff_worktree.patch" 2>/dev/null || true
git diff --cached > "$OUT_DIR/git_diff_staged.patch" 2>/dev/null || true

cp -f README.md "$OUT_DIR/" 2>/dev/null || true
cp -f run_all_local.sh "$OUT_DIR/" 2>/dev/null || true
cp -f setup_local.sh "$OUT_DIR/" 2>/dev/null || true
cp -f auto_improve_loop.sh "$OUT_DIR/" 2>/dev/null || true
cp -f evaluate_review_gate.py "$OUT_DIR/" 2>/dev/null || true
cp -f run_pipeline_local.py "$OUT_DIR/" 2>/dev/null || true
cp -f run_ranker_local.py "$OUT_DIR/" 2>/dev/null || true
cp -f display_report.py "$OUT_DIR/" 2>/dev/null || true

mkdir -p "$OUT_DIR/reports" "$OUT_DIR/logs"
cp -f outputs/reports/latest_detailed_report.json "$OUT_DIR/reports/" 2>/dev/null || true
cp -f outputs/reports/latest_comparison.json "$OUT_DIR/reports/" 2>/dev/null || true
cp -f outputs/reports/latest_gate.json "$OUT_DIR/reports/" 2>/dev/null || true
cp -f outputs/reports/latest_improve_attempts.json "$OUT_DIR/reports/" 2>/dev/null || true

if ls outputs/logs/*.log >/dev/null 2>&1; then
  ls -1t outputs/logs/*.log | head -n 5 | while IFS= read -r f; do
    cp -f "$f" "$OUT_DIR/logs/" 2>/dev/null || true
  done
fi

tar --warning=no-file-changed -czf "${OUT_DIR}.tar.gz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
ln -sfn "$(basename "$OUT_DIR")" outputs/checkpoints/latest

echo "Checkpoint saved: $OUT_DIR"
echo "Archive saved: ${OUT_DIR}.tar.gz"
echo "Latest symlink: outputs/checkpoints/latest"
