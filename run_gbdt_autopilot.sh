#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SYMBOLS="${1:-AAPL,MSFT,NVDA,AMZN,META}"
PREFERRED_MODE="${2:-auto}" # auto | quick | improve
DRY_RUN="${3:-}"            # --dry-run

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

MODE="$(resolve_mode "$PREFERRED_MODE")"

echo "GBDT autopilot"
echo "  symbols: $SYMBOLS"
echo "  requested_mode: $PREFERRED_MODE"
echo "  selected_mode: $MODE"
echo "  dry_run: ${DRY_RUN:-none}"

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  exit 0
fi

./run_gbdt_only.sh "$SYMBOLS" "$MODE"

if [[ -x ./print_latest_kpi.py ]]; then
  python print_latest_kpi.py || true
fi
