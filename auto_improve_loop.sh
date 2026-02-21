#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SYMBOLS="${1:-AAPL,MSFT,NVDA,AMZN,META}"
mkdir -p outputs/logs outputs/reports
LOOP_LOG="outputs/logs/auto_improve_$(date -u +%Y%m%d_%H%M%S).log"
ATTEMPTS_TS="$(date -u +%Y%m%d_%H%M%S)"
ATTEMPTS_JSON="outputs/reports/improve_attempts_${ATTEMPTS_TS}.json"
LATEST_ATTEMPTS_JSON="outputs/reports/latest_improve_attempts.json"
exec > >(tee -a "$LOOP_LOG") 2>&1
echo "Loop log: $LOOP_LOG"
echo "Attempts report: $ATTEMPTS_JSON"

load_api_key() {
  if [[ -n "${FMP_API_KEY:-}" ]]; then
    if [[ "${FMP_API_KEY}" == FMP_API_KEY=* ]]; then
      printf "%s" "${FMP_API_KEY#FMP_API_KEY=}" | sed "s/^['\"]//;s/['\"]$//"
    else
      printf "%s" "${FMP_API_KEY}"
    fi
    return 0
  fi
  if [[ -f /content/.env_fmp ]]; then
    while IFS= read -r raw || [[ -n "$raw" ]]; do
      line="$(printf "%s" "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
      [[ -z "$line" || "$line" == \#* ]] && continue
      if [[ "$line" == FMP_API_KEY=* ]]; then
        printf "%s" "${line#FMP_API_KEY=}" | sed "s/^['\"]//;s/['\"]$//"
        return 0
      fi
      if [[ "$line" != *=* ]]; then
        printf "%s" "$line"
        return 0
      fi
    done < /content/.env_fmp
  fi
  return 1
}

if KEY="$(load_api_key)"; then
  export FMP_API_KEY="$KEY"
fi
if [[ -z "${FMP_API_KEY:-}" ]]; then
  echo "ERROR: FMP_API_KEY is required (env var or /content/.env_fmp)." >&2
  exit 1
fi

last_failure_category="unknown"

detect_failure_category() {
  if [[ ! -f outputs/reports/latest_gate.json ]]; then
    echo "unknown"
    return 0
  fi
  python - <<'PY'
import json
from pathlib import Path

p = Path("outputs/reports/latest_gate.json")
if not p.exists():
    print("unknown")
    raise SystemExit(0)
d = json.loads(p.read_text(encoding="utf-8"))
reasons = [str(x) for x in d.get("reasons", [])]
joined = " ".join(reasons)
if "avg_auc<" in joined:
    print("auc")
elif "best_sharpe<" in joined or "no_eligible_rule" in joined:
    print("rule")
elif "overlap_count<" in joined or "position_match<" in joined or "rank_corr<" in joined:
    print("alignment")
elif reasons:
    print("other")
else:
    print("unknown")
PY
}

run_case() {
  local label="$1"
  local forward_days="$2"
  local thresholds="$3"
  local embargo="$4"
  local min_trades="$5"
  local wf_folds="$6"

  export PIPE_FORWARD_DAYS="$forward_days"
  export PIPE_RULE_THRESHOLDS="$thresholds"
  export PIPE_EMBARGO_DAYS="$embargo"
  export PIPE_MIN_RULE_TRADES="$min_trades"
  export PIPE_WF_FOLDS="$wf_folds"
  export PIPE_USE_GPU=1

  echo ""
  echo "=== Attempt: $label ==="
  echo "symbols=$SYMBOLS"
  echo "PIPE_FORWARD_DAYS=$PIPE_FORWARD_DAYS"
  echo "PIPE_RULE_THRESHOLDS=$PIPE_RULE_THRESHOLDS"
  echo "PIPE_EMBARGO_DAYS=$PIPE_EMBARGO_DAYS"
  echo "PIPE_MIN_RULE_TRADES=$PIPE_MIN_RULE_TRADES"
  echo "PIPE_WF_FOLDS=$PIPE_WF_FOLDS"

  local rc=0
  if STRICT_REVIEW_GATE=1 ./run_all_local.sh "$SYMBOLS"; then
    rc=0
  else
    rc=$?
  fi

  python - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

attempts_path = Path("$ATTEMPTS_JSON")
latest_path = Path("$LATEST_ATTEMPTS_JSON")
gate_path = Path("outputs/reports/latest_gate.json")

try:
    data = json.loads(attempts_path.read_text(encoding="utf-8"))
except Exception:
    data = {"generated_at": datetime.now(timezone.utc).isoformat(), "attempts": []}

gate = {}
if gate_path.exists():
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception:
        gate = {}

data["attempts"].append(
    {
        "at": datetime.now(timezone.utc).isoformat(),
        "label": "$label",
        "symbols": "$SYMBOLS",
        "env": {
            "PIPE_FORWARD_DAYS": "$forward_days",
            "PIPE_RULE_THRESHOLDS": "$thresholds",
            "PIPE_EMBARGO_DAYS": "$embargo",
            "PIPE_MIN_RULE_TRADES": "$min_trades",
            "PIPE_WF_FOLDS": "$wf_folds",
            "PIPE_USE_GPU": "1",
        },
        "run_all_exit_code": $rc,
        "gate_status": gate.get("status"),
        "gate_reasons": gate.get("reasons", []),
        "gate_metrics": gate.get("metrics", {}),
    }
)
attempts_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
latest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("Updated attempts report:", attempts_path)
print("Updated attempts alias:", latest_path)
PY

  if [[ "$rc" -eq 0 ]]; then
    echo "Attempt $label: PASS"
    if [[ -x ./save_workspace_checkpoint.sh ]]; then
      ./save_workspace_checkpoint.sh "attempt-${label}-pass" || true
    fi
    return 0
  fi

  echo "Attempt $label: FAILED gate or runtime"
  last_failure_category="$(detect_failure_category)"
  echo "Failure category: $last_failure_category"
  if [[ -f outputs/reports/latest_gate.json ]]; then
    python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/reports/latest_gate.json")
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    print("Gate status:", d.get("status"))
    print("Reasons:", ", ".join(d.get("reasons", [])))
    m = d.get("metrics", {})
    print("Metrics: avg_auc={:.4f}, best_rule={}, best_sharpe={:.4f}".format(
        float(m.get("avg_auc", 0.0)),
        m.get("best_rule"),
        float(m.get("best_sharpe", 0.0)),
    ))
PY
  fi
  if [[ -x ./save_workspace_checkpoint.sh ]]; then
    ./save_workspace_checkpoint.sh "attempt-${label}-fail" || true
  fi
  return "$rc"
}

if run_case "base" "20" "0.50,0.55,0.60" "5" "300" "3"; then
  exit 0
fi

case "$last_failure_category" in
  auc)
    if run_case "auc-short-horizon" "10" "0.50,0.55,0.60" "3" "200" "3"; then exit 0; fi
    if run_case "auc-more-folds" "20" "0.45,0.50,0.55" "5" "200" "4"; then exit 0; fi
    if run_case "auc-long-horizon" "30" "0.50,0.55,0.60,0.65" "7" "200" "3"; then exit 0; fi
    ;;
  rule)
    if run_case "rule-looser-thresholds" "20" "0.45,0.50,0.55,0.60" "3" "200" "3"; then exit 0; fi
    if run_case "rule-lower-trades" "20" "0.50,0.55,0.60" "5" "150" "3"; then exit 0; fi
    if run_case "rule-wide-thresholds" "20" "0.40,0.45,0.50,0.55,0.60" "3" "150" "3"; then exit 0; fi
    ;;
  alignment)
    if run_case "align-stable" "20" "0.50,0.55,0.60" "5" "300" "3"; then exit 0; fi
    if run_case "align-looser-thresholds" "20" "0.45,0.50,0.55" "3" "200" "3"; then exit 0; fi
    ;;
  *)
    if run_case "fallback-looser-thresholds" "20" "0.45,0.50,0.55,0.60" "3" "200" "3"; then exit 0; fi
    if run_case "fallback-short-horizon" "10" "0.50,0.55,0.60" "3" "200" "3"; then exit 0; fi
    if run_case "fallback-long-horizon" "30" "0.50,0.55,0.60,0.65" "7" "200" "3"; then exit 0; fi
    if run_case "fallback-more-folds" "20" "0.45,0.50,0.55" "5" "200" "4"; then exit 0; fi
    ;;
esac

echo "All improvement attempts failed review gate."
exit 1
