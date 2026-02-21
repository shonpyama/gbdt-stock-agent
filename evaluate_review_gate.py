#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_gate(
    detail: dict,
    comp: dict,
    min_avg_auc: float,
    min_best_sharpe: float,
    min_overlap: int,
    min_position_match: int,
    min_rank_corr: float,
) -> dict:
    wf = detail.get("walk_forward", {})
    rules = detail.get("exit_rules_comparison", {})

    best_rule = None
    best_sharpe = float("-inf")
    for name, r in rules.items():
        if not bool(r.get("eligible", True)):
            continue
        s = float(r.get("sharpe_net", 0.0))
        if s > best_sharpe:
            best_sharpe = s
            best_rule = name

    needs_review = False
    reasons: list[str] = []
    avg_auc = float(wf.get("avg_auc", 0.0))
    if avg_auc < min_avg_auc:
        needs_review = True
        reasons.append(f"avg_auc<{min_avg_auc:.2f} ({avg_auc:.4f})")

    if best_rule is None:
        needs_review = True
        reasons.append("no_eligible_rule")
        best_sharpe = 0.0
    elif best_sharpe < min_best_sharpe:
        needs_review = True
        reasons.append(f"best_sharpe<{min_best_sharpe:.2f} ({best_sharpe:.4f})")

    cmp_used = False
    cmp_date_d = comp.get("pipeline_date")
    cmp_date_r = comp.get("ranker_date")
    overlap_count = comp.get("overlap_count")
    position_match = comp.get("position_match")
    rank_corr = comp.get("rank_corr")
    if cmp_date_d and cmp_date_r and cmp_date_d == cmp_date_r:
        cmp_used = True
        overlap_i = int(overlap_count or 0)
        if overlap_i < min_overlap:
            needs_review = True
            reasons.append(f"overlap_count<{min_overlap} ({overlap_i})")
        if position_match is not None and int(position_match) < min_position_match:
            needs_review = True
            reasons.append(f"position_match<{min_position_match} ({position_match})")
        if rank_corr is not None and float(rank_corr) < min_rank_corr:
            needs_review = True
            reasons.append(f"rank_corr<{min_rank_corr:.2f} ({float(rank_corr):.3f})")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "needs_review": needs_review,
        "status": "REVIEW_REQUIRED" if needs_review else "PASS",
        "reasons": reasons,
        "thresholds": {
            "min_avg_auc": min_avg_auc,
            "min_best_sharpe": min_best_sharpe,
            "min_overlap": min_overlap,
            "min_position_match": min_position_match,
            "min_rank_corr": min_rank_corr,
        },
        "metrics": {
            "avg_auc": avg_auc,
            "best_rule": best_rule,
            "best_sharpe": best_sharpe,
            "comparison_used": cmp_used,
            "overlap_count": overlap_count,
            "position_match": position_match,
            "rank_corr": rank_corr,
            "pipeline_date": cmp_date_d,
            "ranker_date": cmp_date_r,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate review gate from report json.")
    parser.add_argument(
        "--detailed-report",
        default="outputs/reports/latest_detailed_report.json",
        help="Path to detailed pipeline report JSON",
    )
    parser.add_argument(
        "--comparison-report",
        default="outputs/reports/latest_comparison.json",
        help="Path to comparison report JSON",
    )
    parser.add_argument(
        "--out",
        default="outputs/reports/latest_gate.json",
        help="Path to gate result JSON",
    )
    parser.add_argument("--min-avg-auc", type=float, default=0.50)
    parser.add_argument("--min-best-sharpe", type=float, default=0.30)
    parser.add_argument("--min-overlap", type=int, default=3)
    parser.add_argument("--min-position-match", type=int, default=2)
    parser.add_argument("--min-rank-corr", type=float, default=0.20)
    parser.add_argument("--strict", action="store_true", help="Exit 1 if gate fails")
    args = parser.parse_args()

    detail_path = Path(args.detailed_report)
    comp_path = Path(args.comparison_report)
    out_path = Path(args.out)

    if not detail_path.exists():
        print(f"ERROR: detailed report not found: {detail_path}", file=sys.stderr)
        return 2
    detail = load_json(detail_path)
    comp = load_json(comp_path) if comp_path.exists() else {}

    result = evaluate_gate(
        detail=detail,
        comp=comp,
        min_avg_auc=args.min_avg_auc,
        min_best_sharpe=args.min_best_sharpe,
        min_overlap=args.min_overlap,
        min_position_match=args.min_position_match,
        min_rank_corr=args.min_rank_corr,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if result["needs_review"]:
        print(f"Review gate: REVIEW REQUIRED ({', '.join(result['reasons'])})")
    else:
        print("Review gate: PASS")
    print(f"Saved gate result: {out_path}")

    if args.strict and result["needs_review"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
