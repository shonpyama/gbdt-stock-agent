#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REMOTE="${1:-origin}"
BRANCH="${2:-main}"
COMMIT="${3:-HEAD}"
TMP_BRANCH="push_tmp_$(date -u +%Y%m%d_%H%M%S)"
STASH_MSG="temp-before-safe-push-${TMP_BRANCH}"

echo "Safe push:"
echo "  remote=$REMOTE"
echo "  branch=$BRANCH"
echo "  commit=$COMMIT"

if ! git rev-parse --verify "$COMMIT" >/dev/null 2>&1; then
  echo "ERROR: commit not found: $COMMIT" >&2
  exit 1
fi

had_changes=0
if [[ -n "$(git status --porcelain)" ]]; then
  had_changes=1
  git stash push -u -m "$STASH_MSG" >/dev/null
  echo "Stashed working changes: $STASH_MSG"
fi

cleanup() {
  local rc=$?
  set +e
  if [[ -f .git/CHERRY_PICK_HEAD ]]; then
    git cherry-pick --abort >/dev/null 2>&1 || true
  fi
  if [[ -d .git/rebase-merge || -d .git/rebase-apply ]]; then
    git rebase --abort >/dev/null 2>&1 || true
  fi
  git checkout main >/dev/null 2>&1 || true
  if [[ "$had_changes" -eq 1 ]]; then
    local stash_ref
    stash_ref="$(git stash list | awk -v msg="$STASH_MSG" '$0 ~ msg {print $1; exit}')"
    if [[ -n "$stash_ref" ]]; then
      git stash pop "$stash_ref" >/dev/null 2>&1 || true
      echo "Restored stashed working changes."
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT

if [[ -f .git/CHERRY_PICK_HEAD ]]; then
  git cherry-pick --abort >/dev/null 2>&1 || true
fi
if [[ -d .git/rebase-merge || -d .git/rebase-apply ]]; then
  git rebase --abort >/dev/null 2>&1 || true
fi

git fetch "$REMOTE" "$BRANCH"
git checkout -B "$TMP_BRANCH" "$REMOTE/$BRANCH" >/dev/null
if git merge-base --is-ancestor "$COMMIT" "$REMOTE/$BRANCH"; then
  echo "Commit $COMMIT is already reachable from $REMOTE/$BRANCH. Nothing to push."
  exit 0
fi
git cherry-pick "$COMMIT"
if [[ -f .git/CHERRY_PICK_HEAD ]]; then
  if git diff --cached --quiet; then
    git cherry-pick --skip >/dev/null 2>&1 || true
  fi
fi
git push "$REMOTE" "$TMP_BRANCH:$BRANCH"
echo "Pushed $COMMIT to $REMOTE/$BRANCH via $TMP_BRANCH"
