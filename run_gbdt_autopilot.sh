#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SYMBOLS="${1:-AAPL,MSFT,NVDA,AMZN,META}"
PREFERRED_MODE="${2:-auto}" # auto | quick | improve
DRY_RUN="${3:-}"            # --dry-run
USE_REVIEW_SIGNAL="${USE_REVIEW_SIGNAL:-1}"
REVIEW_WINDOW="${REVIEW_WINDOW:-5}"
REVIEW_MIN_AUC="${REVIEW_MIN_AUC:-0.52}"
REVIEW_MAX_AUC_DROP="${REVIEW_MAX_AUC_DROP:-0.03}"

resolve_mode() {
  local requested="$1"
  if [[ "$requested" == "quick" || "$requested" == "improve" ]]; then
    echo "$requested"
    return 0
  fi
  if [[ ! -f outputs/reports/latest_gate.json ]]; then
    echo "improve"
    return 0
  fi
  if [[ "$USE_REVIEW_SIGNAL" == "1" && -f outputs/reports/latest_review_signal.json ]]; then
    local signal
    signal="$(python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/reports/latest_review_signal.json")
if not p.exists():
    print("")
else:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        print((d.get("status") or "").upper())
    except Exception:
        print("")
PY
)"
    if [[ "$signal" == "REVIEW_REQUIRED" ]]; then
      echo "improve"
      return 0
    fi
  fi
  python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/reports/latest_gate.json")
if not p.exists():
    print("improve")
    raise SystemExit(0)
try:
    d = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print("improve")
    raise SystemExit(0)
print("quick" if d.get("status") == "PASS" else "improve")
PY
}

echo "GBDT autopilot"
echo "  symbols: $SYMBOLS"
echo "  requested_mode: $PREFERRED_MODE"
echo "  dry_run: ${DRY_RUN:-none}"
echo "  use_review_signal: $USE_REVIEW_SIGNAL"

if [[ -f ./analyze_recent_runs.py ]]; then
  python analyze_recent_runs.py \
    --window "$REVIEW_WINDOW" \
    --min-auc "$REVIEW_MIN_AUC" \
    --max-auc-drop "$REVIEW_MAX_AUC_DROP" || true
fi

MODE="$(resolve_mode "$PREFERRED_MODE")"
echo "  selected_mode: $MODE"

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  exit 0
fi

./run_gbdt_only.sh "$SYMBOLS" "$MODE"

if [[ -x ./print_latest_kpi.py ]]; then
  python print_latest_kpi.py || true
fi

if [[ -f ./analyze_recent_runs.py ]]; then
  python analyze_recent_runs.py \
    --window "$REVIEW_WINDOW" \
    --min-auc "$REVIEW_MIN_AUC" \
    --max-auc-drop "$REVIEW_MAX_AUC_DROP" || true
fi
