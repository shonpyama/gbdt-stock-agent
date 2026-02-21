#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SYMBOLS="${1:-AAPL,MSFT,NVDA,AMZN,META}"
mkdir -p outputs/logs
LOG_PATH="outputs/logs/run_all_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_PATH") 2>&1
echo "Log file: $LOG_PATH"

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

echo "[1/3] Running detailed pipeline for: $SYMBOLS"
PIPE_OUTPUT="$(python run_pipeline_local.py "$SYMBOLS")"
echo "$PIPE_OUTPUT"
DETAIL_REPORT="$(echo "$PIPE_OUTPUT" | sed -n 's/^Saved report: //p' | tail -n 1)"
PIPELINE_DATE=""
if [[ -n "$DETAIL_REPORT" && -f "$DETAIL_REPORT" ]]; then
  PIPELINE_DATE="$(python - "$DETAIL_REPORT" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("top_candidates", [{}])[0].get("date", ""))
PY
)"
  PIPELINE_DATE="$(echo "$PIPELINE_DATE" | tr -d '\r\n')"
fi

echo "[2/3] Running ranker for: $SYMBOLS"
if [[ -n "$PIPELINE_DATE" ]]; then
  echo "Aligning ranker as-of date to pipeline date: $PIPELINE_DATE"
  RANKER_OUTPUT="$(RANKER_ASOF_DATE="$PIPELINE_DATE" python run_ranker_local.py "$SYMBOLS")"
else
  RANKER_OUTPUT="$(python run_ranker_local.py "$SYMBOLS")"
fi
echo "$RANKER_OUTPUT"
RANKER_REPORT="$(echo "$RANKER_OUTPUT" | sed -n 's/^   JSON: //p' | tail -n 1)"

echo "[3/4] Comparing top candidates (pipeline vs ranker)"
if [[ -n "$DETAIL_REPORT" && -f "$DETAIL_REPORT" && -n "$RANKER_REPORT" && -f "$RANKER_REPORT" ]]; then
  python - "$DETAIL_REPORT" "$RANKER_REPORT" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
with open(sys.argv[2], "r", encoding="utf-8") as f:
    r = json.load(f)

top_d = [x.get("symbol") for x in d.get("top_candidates", [])[:5]]
top_r = [x.get("symbol") for x in r.get("top_candidates", [])[:5]]
overlap = [s for s in top_d if s in top_r]
date_d = d.get("top_candidates", [{}])[0].get("date")
date_r = r.get("top_candidates", [{}])[0].get("date")
position_match = None
rank_corr = None
print(f"Pipeline Top5: {top_d}")
print(f"Ranker   Top5: {top_r}")
print(f"Overlap count: {len(overlap)}/5 ({overlap})")
print(f"Pipeline date: {date_d}")
print(f"Ranker   date: {date_r}")
if date_d and date_r and date_d == date_r:
    pos_match = sum(1 for i, s in enumerate(top_d) if i < len(top_r) and top_r[i] == s)
    rank_r = {s: i for i, s in enumerate(top_r)}
    common = [s for s in top_d if s in rank_r]
    if len(common) >= 2:
        d_pos = [top_d.index(s) for s in common]
        r_pos = [rank_r[s] for s in common]
        d_center = sum(d_pos) / len(d_pos)
        r_center = sum(r_pos) / len(r_pos)
        cov = sum((a - d_center) * (b - r_center) for a, b in zip(d_pos, r_pos))
        d_var = sum((a - d_center) ** 2 for a in d_pos)
        r_var = sum((b - r_center) ** 2 for b in r_pos)
        rank_corr = cov / ((d_var * r_var) ** 0.5) if d_var > 0 and r_var > 0 else 0.0
    else:
        rank_corr = 0.0
    print(f"Position match: {pos_match}/5")
    print(f"Rank corr (Top5 common): {rank_corr:.3f}")
    position_match = pos_match
else:
    print("Position/rank comparison skipped: different ranking dates.")

out = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "pipeline_report": sys.argv[1],
    "ranker_report": sys.argv[2],
    "pipeline_top5": top_d,
    "ranker_top5": top_r,
    "overlap_count": len(overlap),
    "pipeline_date": date_d,
    "ranker_date": date_r,
    "position_match": position_match,
    "rank_corr": rank_corr,
}
os.makedirs("outputs/reports", exist_ok=True)
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
path = f"outputs/reports/comparison_{ts}.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
with open("outputs/reports/latest_comparison.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Saved comparison: {path}")
print("Saved comparison alias: outputs/reports/latest_comparison.json")
PY
else
  echo "Compare skipped: report path missing."
fi

echo "[4/4] Displaying detailed pipeline report"
if [[ -n "$DETAIL_REPORT" && -f "$DETAIL_REPORT" ]]; then
  cp "$DETAIL_REPORT" outputs/reports/latest_detailed_report.json
  python - "$DETAIL_REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)

wf = d.get("walk_forward", {})
rules = d.get("exit_rules_comparison", {})
meta = d.get("meta", {})
score_def = meta.get("score_definition", {})
run_cfg = meta.get("run_config", {})
cmp = {}
try:
    with open("outputs/reports/latest_comparison.json", "r", encoding="utf-8") as f:
        cmp = json.load(f)
except Exception:
    cmp = {}
best_rule = None
best_sharpe = float("-inf")
for name, r in rules.items():
    if not bool(r.get("eligible", True)):
        continue
    s = float(r.get("sharpe_net", 0.0))
    if s > best_sharpe:
        best_sharpe = s
        best_rule = name

print("KPI Summary:")
print(f"  WF avg acc: {wf.get('avg_accuracy', 0):.4f}")
print(f"  WF avg auc: {wf.get('avg_auc', 0):.4f}")
print(f"  Direction mode: {score_def.get('direction_mode', 'n/a')}")
print(f"  Device holdout/live: {run_cfg.get('train_device_holdout', 'n/a')}/{run_cfg.get('train_device_live', 'n/a')}")
if cmp:
    print(
        "  Compare overlap/pos/rankcorr: "
        f"{cmp.get('overlap_count', 'n/a')}/5, "
        f"{cmp.get('position_match', 'n/a')}, "
        f"{cmp.get('rank_corr', 'n/a')}"
    )
if best_rule is not None:
    br = rules[best_rule]
    print(f"  Best rule: {best_rule}")
    print(f"  Rule sharpe: {float(br.get('sharpe_net', 0.0)):.4f}")
    print(f"  Rule win rate: {float(br.get('win_rate_pct', 0.0)):.2f}%")
else:
    print("  Best rule: None (all rules ineligible)")
print("  Latest report alias: outputs/reports/latest_detailed_report.json")

needs_review = False
review_reasons = []
avg_auc = float(wf.get("avg_auc", 0.0))
if avg_auc < 0.50:
    needs_review = True
    review_reasons.append(f"avg_auc<{0.50:.2f} ({avg_auc:.4f})")
if best_rule is None:
    needs_review = True
    review_reasons.append("no_eligible_rule")
else:
    best_sharpe = float(br.get("sharpe_net", 0.0))
    if best_sharpe < 0.30:
        needs_review = True
        review_reasons.append(f"best_sharpe<{0.30:.2f} ({best_sharpe:.4f})")

cmp_date_d = cmp.get("pipeline_date")
cmp_date_r = cmp.get("ranker_date")
if cmp_date_d and cmp_date_r and cmp_date_d == cmp_date_r:
    top_d = cmp.get("pipeline_top5") or []
    top_r = cmp.get("ranker_top5") or []
    effective_min_overlap = min(3, len(top_d), len(top_r))
    overlap_count = int(cmp.get("overlap_count", 0))
    pos_match = cmp.get("position_match")
    rank_corr = cmp.get("rank_corr")
    if overlap_count < effective_min_overlap:
        needs_review = True
        review_reasons.append(f"overlap_count<{effective_min_overlap} ({overlap_count})")
    if pos_match is not None and pos_match < 2:
        needs_review = True
        review_reasons.append(f"position_match<2 ({pos_match})")
    if rank_corr is not None and float(rank_corr) < 0.20:
        needs_review = True
        review_reasons.append(f"rank_corr<0.20 ({float(rank_corr):.3f})")

if needs_review:
    print("  REVIEW REQUIRED: " + ", ".join(review_reasons))
else:
    print("  Review gate: PASS")
PY
  python display_report.py "$DETAIL_REPORT"
else
  python display_report.py
fi

echo "[5/5] Evaluating review gate artifact"
STRICT_GATE="${STRICT_REVIEW_GATE:-0}"
GATE_MIN_AVG_AUC="${GATE_MIN_AVG_AUC:-0.50}"
GATE_MIN_BEST_SHARPE="${GATE_MIN_BEST_SHARPE:-0.30}"
GATE_MIN_OVERLAP="${GATE_MIN_OVERLAP:-3}"
GATE_MIN_POSITION_MATCH="${GATE_MIN_POSITION_MATCH:-2}"
GATE_MIN_RANK_CORR="${GATE_MIN_RANK_CORR:-0.20}"
GATE_CMD=(python evaluate_review_gate.py
  --detailed-report outputs/reports/latest_detailed_report.json
  --comparison-report outputs/reports/latest_comparison.json
  --out outputs/reports/latest_gate.json
  --min-avg-auc "$GATE_MIN_AVG_AUC"
  --min-best-sharpe "$GATE_MIN_BEST_SHARPE"
  --min-overlap "$GATE_MIN_OVERLAP"
  --min-position-match "$GATE_MIN_POSITION_MATCH"
  --min-rank-corr "$GATE_MIN_RANK_CORR"
)
if [[ "$STRICT_GATE" == "1" ]]; then
  echo "Strict review gate enabled."
  GATE_CMD+=(--strict)
fi
GATE_RC=0
set +e
"${GATE_CMD[@]}"
GATE_RC=$?
set -e

echo "[6/6] Writing run manifest"
python - "$DETAIL_REPORT" "$RANKER_REPORT" "$LOG_PATH" "$SYMBOLS" "$STRICT_GATE" "$GATE_RC" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

detail_report = sys.argv[1]
ranker_report = sys.argv[2]
log_path = sys.argv[3]
symbols = sys.argv[4]
strict_gate = sys.argv[5]
gate_rc = int(sys.argv[6])

def git_out(args: list[str]) -> str:
    try:
        return (
            subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
    except Exception:
        return ""

gate_data = {}
gate_path = Path("outputs/reports/latest_gate.json")
if gate_path.exists():
    try:
        gate_data = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception:
        gate_data = {}

manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "symbols": symbols,
    "strict_review_gate": strict_gate == "1",
    "gate_exit_code": gate_rc,
    "gate_status": gate_data.get("status"),
    "gate_reasons": gate_data.get("reasons", []),
    "artifacts": {
        "detailed_report": detail_report if detail_report else "",
        "ranker_report": ranker_report if ranker_report else "",
        "comparison_report": "outputs/reports/latest_comparison.json",
        "gate_report": "outputs/reports/latest_gate.json",
        "log_path": log_path,
    },
    "git": {
        "branch": git_out(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": git_out(["rev-parse", "HEAD"]),
        "status_short": git_out(["status", "--short"]),
    },
}

out_dir = Path("outputs/reports")
out_dir.mkdir(parents=True, exist_ok=True)
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
manifest_path = out_dir / f"run_manifest_{ts}.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
(out_dir / "latest_run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Saved run manifest: {manifest_path}")
print("Saved run manifest alias: outputs/reports/latest_run_manifest.json")
PY

if [[ "$STRICT_GATE" == "1" && "$GATE_RC" -ne 0 ]]; then
  exit "$GATE_RC"
fi
