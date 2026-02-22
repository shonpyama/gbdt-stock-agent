#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _mean(xs: Iterable[float]) -> float:
    vals = [x for x in xs if x == x]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _parse_inputs(raw: str) -> List[Path]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("--inputs is required and must contain at least one path")
    return [Path(p) for p in parts]


def _detect_mode(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("No rows to detect mode")
    r0 = rows[0]
    if "params" in r0:
        return "model"
    if "lookbacks" in r0 and "event_shift" in r0:
        return "feature"
    raise ValueError("Failed to detect mode from results rows")


def _dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # When reruns exist for same (end_date, name), keep the better scored row.
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        key = (str(r.get("end_date")), str(r.get("name")))
        old = best.get(key)
        if old is None:
            best[key] = r
            continue
        if _safe_float(r.get("score")) > _safe_float(old.get("score")):
            best[key] = r
    return list(best.values())


def _summarize(rows: List[Dict[str, Any]], end_dates: List[str], mode: str) -> List[Dict[str, Any]]:
    names = sorted({str(r.get("name")) for r in rows})
    summary: List[Dict[str, Any]] = []
    for name in names:
        group = [r for r in rows if str(r.get("name")) == name]
        base: Dict[str, Any] = {
            "name": name,
            "periods": len(group),
            "gate_pass_periods": sum(1 for r in group if bool(r.get("ops_gate_ok"))),
            "all_periods_gate_pass": all(bool(r.get("ops_gate_ok")) for r in group) and len(group) == len(end_dates),
            "score_mean": _mean([_safe_float(r.get("score")) for r in group]),
            "score_min": min((_safe_float(r.get("score")) for r in group), default=float("nan")),
            "rank_ic_mean": _mean([_safe_float(r.get("rank_ic_test_mean")) for r in group]),
            "sharpe_mean": _mean([_safe_float(r.get("sharpe")) for r in group]),
            "total_return_mean": _mean([_safe_float(r.get("total_return")) for r in group]),
            "max_drawdown_mean": _mean([_safe_float(r.get("max_drawdown")) for r in group]),
        }
        r0 = group[0] if group else {}
        if mode == "model":
            base["params"] = dict(r0.get("params") or {})
        else:
            base["lookbacks"] = list(r0.get("lookbacks") or [])
            base["event_shift"] = int(r0.get("event_shift") or 1)
        summary.append(base)
    return sorted(
        summary,
        key=lambda x: (
            1 if bool(x.get("all_periods_gate_pass")) else 0,
            _safe_float(x.get("score_mean")),
            _safe_float(x.get("score_min")),
        ),
        reverse=True,
    )


def _render_md(
    mode: str,
    base_conf: str,
    policy_path: str,
    end_dates: List[str],
    rows: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
    selected: Dict[str, Any] | None,
) -> str:
    title = "Model Stability Merge" if mode == "model" else "Feature Stability Merge"
    lines = [
        f"# {title} - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        f"- mode: `{mode}`",
        f"- base_conf: `{base_conf}`",
        f"- policy: `{policy_path}`",
        f"- end_dates: `{json.dumps(end_dates, ensure_ascii=True)}`",
        "",
        "## Per-Period",
        "",
        "| end_date | name | score | rank_ic_test_mean | sharpe | total_return | max_drawdown | ops_gate | run_id |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (str(x.get("end_date")), str(x.get("name")))):
        gate_txt = "pass" if bool(r.get("ops_gate_ok")) else "fail"
        lines.append(
            f"| {r.get('end_date')} | {r.get('name')} | {r.get('score')} | {r.get('rank_ic_test_mean')} | {r.get('sharpe')} | {r.get('total_return')} | {r.get('max_drawdown')} | {gate_txt} | {r.get('run_id','-')} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        "| rank | name | all_periods_gate_pass | score_mean | score_min | rank_ic_mean | sharpe_mean | total_return_mean | max_drawdown_mean |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, s in enumerate(summary, start=1):
        lines.append(
            f"| {idx} | {s.get('name')} | {s.get('all_periods_gate_pass')} | {s.get('score_mean')} | {s.get('score_min')} | {s.get('rank_ic_mean')} | {s.get('sharpe_mean')} | {s.get('total_return_mean')} | {s.get('max_drawdown_mean')} |"
        )
    if selected:
        lines += [
            "",
            "## Selected",
            "",
            f"- name: `{selected.get('name')}`",
            f"- score_mean: `{selected.get('score_mean')}`",
            f"- score_min: `{selected.get('score_min')}`",
        ]
        if mode == "model":
            lines.append(f"- params: `{json.dumps(selected.get('params', {}), ensure_ascii=True)}`")
        else:
            lines.append(f"- lookbacks: `{json.dumps(selected.get('lookbacks', []), ensure_ascii=True)}`")
            lines.append(f"- event_shift: `{selected.get('event_shift')}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sharded model/feature stability result JSON files.")
    parser.add_argument("--inputs", required=True, help="Comma-separated result JSON paths.")
    parser.add_argument("--out-json", required=True, help="Output merged JSON path.")
    parser.add_argument("--out-md", required=True, help="Output merged Markdown path.")
    args = parser.parse_args()

    inputs = _parse_inputs(args.inputs)
    payloads: List[Dict[str, Any]] = []
    for p in inputs:
        payloads.append(json.loads(p.read_text()))

    all_rows: List[Dict[str, Any]] = []
    for pl in payloads:
        all_rows.extend(list(pl.get("results") or []))
    rows = _dedup_rows(all_rows)
    mode = _detect_mode(rows)

    end_dates = sorted({str(r.get("end_date")) for r in rows})
    base_conf = str((payloads[0].get("base_conf") if payloads else "") or "")
    policy_path = str((payloads[0].get("policy_path") if payloads else "") or "")
    summary = _summarize(rows=rows, end_dates=end_dates, mode=mode)
    selected = summary[0] if summary else None

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    merged = {
        "mode": mode,
        "merged_from": [str(p) for p in inputs],
        "base_conf": base_conf,
        "policy_path": policy_path,
        "end_dates": end_dates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": rows,
        "summary": summary,
        "selected": selected,
    }
    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=True))
    out_md.write_text(
        _render_md(
            mode=mode,
            base_conf=base_conf,
            policy_path=policy_path,
            end_dates=end_dates,
            rows=rows,
            summary=summary,
            selected=selected,
        )
    )
    print(
        json.dumps(
            {
                "mode": mode,
                "inputs": [str(p) for p in inputs],
                "out_json": str(out_json),
                "out_md": str(out_md),
                "selected": selected,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
