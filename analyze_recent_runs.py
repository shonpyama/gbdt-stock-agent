#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_best_sharpe(detail: dict) -> float:
    rules = detail.get("exit_rules_comparison", {})
    best = float("-inf")
    for _, r in rules.items():
        if not bool(r.get("eligible", True)):
            continue
        best = max(best, float(r.get("sharpe_net", 0.0)))
    if best == float("-inf"):
        return 0.0
    return best


def is_stress_gate(reasons: list[str]) -> bool:
    for r in reasons:
        m_auc = re.search(r"avg_auc<([0-9.]+)", r)
        if m_auc:
            try:
                if float(m_auc.group(1)) > 0.70:
                    return True
            except Exception:
                pass
        m_sharpe = re.search(r"best_sharpe<([0-9.]+)", r)
        if m_sharpe:
            try:
                if float(m_sharpe.group(1)) > 1.00:
                    return True
            except Exception:
                pass
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze recent GBDT run quality trend.")
    parser.add_argument("--window", type=int, default=5, help="Number of recent manifests")
    parser.add_argument("--min-auc", type=float, default=0.52)
    parser.add_argument("--max-auc-drop", type=float, default=0.03)
    parser.add_argument("--include-stress", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    reports_dir = Path("outputs/reports")
    manifests = sorted(
        reports_dir.glob("run_manifest_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    manifests = manifests[-max(1, args.window) :]

    rows = []
    skipped = 0
    review_required_count = 0
    for m in manifests:
        md = load_json(m)
        gate_status = (md.get("gate_status") or "").upper()
        gate_reasons = [str(x) for x in md.get("gate_reasons", [])]
        if not args.include_stress and is_stress_gate(gate_reasons):
            skipped += 1
            continue
        if gate_status == "REVIEW_REQUIRED":
            review_required_count += 1

        dpath = md.get("artifacts", {}).get("detailed_report", "")
        detail = load_json(Path(dpath)) if dpath else {}
        wf = detail.get("walk_forward", {})
        auc = float(wf.get("avg_auc", 0.0))
        sharpe = extract_best_sharpe(detail)
        rows.append(
            {
                "manifest": str(m),
                "generated_at": md.get("generated_at"),
                "gate_status": gate_status or "UNKNOWN",
                "gate_reasons": gate_reasons,
                "avg_auc": auc,
                "best_sharpe": sharpe,
            }
        )

    needs_review = False
    reasons: list[str] = []
    if review_required_count > 0:
        needs_review = True
        reasons.append(f"recent_review_required={review_required_count}")

    auc_values = [r["avg_auc"] for r in rows if r["avg_auc"] > 0]
    if auc_values:
        last_auc = auc_values[-1]
        if last_auc < args.min_auc:
            needs_review = True
            reasons.append(f"last_auc<{args.min_auc:.2f} ({last_auc:.4f})")
        if len(auc_values) >= 2:
            auc_drop = auc_values[0] - auc_values[-1]
            if auc_drop > args.max_auc_drop:
                needs_review = True
                reasons.append(
                    f"auc_drop>{args.max_auc_drop:.2f} ({auc_drop:.4f})"
                )

    out = {
        "window": args.window,
        "needs_review": needs_review,
        "status": "REVIEW_REQUIRED" if needs_review else "PASS",
        "reasons": reasons,
        "runs_analyzed": len(rows),
        "runs_skipped_as_stress": skipped,
        "rows": rows,
    }

    out_path = reports_dir / "latest_review_signal.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Recent run signal:", out["status"])
    if reasons:
        print("Reasons:", ", ".join(reasons))
    print(f"Saved signal: {out_path}")

    if args.strict and needs_review:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
