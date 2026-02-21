from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .colab import restore_runtime_from_drive, sync_runtime_to_drive
from .config import load_config
from .fmp_client import FMPClient, cache_key, resolve_fmp_api_key
from .migrate import pack_bundle, restore_bundle
from .operations import collect_ops_status, evaluate_ops_gate, load_ops_policy, write_ops_incident, write_ops_snapshot
from .orchestrator import run_pipeline
from .paths import ProjectPaths
from .reporting import render_report_md, write_report
from .transition import generate_transition_report


def _project_dir_from_cwd() -> Path:
    return Path.cwd()


def _load_cfg(conf_path: str) -> Dict[str, Any]:
    return load_config(Path(conf_path))


def cmd_preflight(args: argparse.Namespace) -> int:
    project_dir = _project_dir_from_cwd()
    paths = ProjectPaths.from_project_dir(project_dir)
    paths.ensure_base_dirs()
    cfg = _load_cfg(args.conf)

    out: Dict[str, Any] = {
        "ok": False,
        "checks": {
            "api_key_env": False,
            "endpoint_call": False,
            "cache_write": False,
        },
    }

    try:
        import os

        api_key = resolve_fmp_api_key()
        out["checks"]["api_key_env"] = bool(api_key)
        if not out["checks"]["api_key_env"]:
            raise RuntimeError("Missing FMP_API_KEY")

        fmp = FMPClient.from_env(paths.cache_http_dir, cfg_overrides=cfg.get("fmp", {}))
        sym = str(args.symbol)
        start = (datetime.now(timezone.utc).date() - timedelta(days=int(args.lookback_days))).isoformat()

        ep = fmp.endpoint_for("prices_eod")
        url_path = f"{fmp.cfg.base_url.rstrip('/')}/{ep}"
        key = cache_key("GET", url_path, {"symbol": sym, "from": start})
        cache_path = paths.cache_http_dir / f"{key}.json"

        _ = fmp.get_prices(sym, start, None)
        out["checks"]["endpoint_call"] = True
        out["checks"]["cache_write"] = cache_path.exists()
        out["cache_path"] = str(cache_path)
        out["ok"] = bool(all(out["checks"].values()))
        print(json.dumps(out, indent=2, ensure_ascii=True))
        return 0 if out["ok"] else 1
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        key = resolve_fmp_api_key() or ""
        if key:
            msg = msg.replace(key, "****")
        out["error"] = msg
        print(json.dumps(out, indent=2, ensure_ascii=True))
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    project_dir = _project_dir_from_cwd()
    run_id = run_pipeline(
        project_dir=project_dir,
        conf_path=Path(args.conf),
        resume=bool(args.resume),
        force_stage=args.force_stage,
        force_unlock=bool(args.force_unlock),
        stop_after_stage=args.stop_after_stage,
    )
    print(json.dumps({"run_id": run_id}, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    project_dir = _project_dir_from_cwd()
    paths = ProjectPaths.from_project_dir(project_dir)
    run_dir = paths.run_dir(args.run_id)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(str(metrics_path))
    metrics = json.loads(metrics_path.read_text())

    cfg = _load_cfg(args.conf) if args.conf else load_config(run_dir / "config.yaml")
    report = render_report_md(
        cfg=cfg,
        run_id=args.run_id,
        conf_hash=str(metrics.get("conf_hash", "")),
        dataset_id=metrics.get("dataset_id"),
        feature_store_id=metrics.get("feature_store_id"),
        code_hash=metrics.get("code_hash"),
        universe_info=metrics.get("universe"),
        split_info=metrics.get("split_info"),
        validation=metrics.get("validation"),
        leakage=metrics.get("leakage"),
        training_info=metrics.get("training_info"),
        model_metrics=metrics.get("model_metrics"),
        chosen_model=(metrics.get("backtest") or {}).get("chosen_model"),
        backtest_summary=(metrics.get("backtest") or {}).get("summary"),
        status=str(metrics.get("status", "unknown")),
        errors=metrics.get("errors"),
        historical_errors=metrics.get("historical_errors"),
    )
    out = run_dir / "report.md"
    write_report(out, report)
    print(str(out))
    return 0


def cmd_transition_report(args: argparse.Namespace) -> int:
    project_dir = _project_dir_from_cwd()
    out = generate_transition_report(project_dir=project_dir, run_id=args.run_id, target=args.target)
    print(str(out))
    return 0


def cmd_migrate_pack(args: argparse.Namespace) -> int:
    project_dir = _project_dir_from_cwd()
    result = pack_bundle(project_dir=project_dir, run_id=args.run_id, out_zip=Path(args.out))
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


def cmd_migrate_restore(args: argparse.Namespace) -> int:
    project_dir = _project_dir_from_cwd()
    result = restore_bundle(archive=Path(args.archive), project_dir=project_dir)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


def cmd_colab_restore(args: argparse.Namespace) -> int:
    stats = restore_runtime_from_drive(drive_path=Path(args.drive_path) if args.drive_path else None)
    print(json.dumps(stats, indent=2, ensure_ascii=True))
    return 0


def cmd_colab_sync(args: argparse.Namespace) -> int:
    stats = sync_runtime_to_drive(drive_path=Path(args.drive_path) if args.drive_path else None)
    print(json.dumps(stats, indent=2, ensure_ascii=True))
    return 0


def cmd_ops_status(args: argparse.Namespace) -> int:
    project_dir = _project_dir_from_cwd()
    payload = collect_ops_status(
        project_dir=project_dir,
        run_id=args.run_id,
        max_age_hours=float(args.max_age_hours),
        require_gpu=bool(args.require_gpu),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_ops_snapshot(args: argparse.Namespace) -> int:
    project_dir = _project_dir_from_cwd()
    payload = collect_ops_status(
        project_dir=project_dir,
        run_id=args.run_id,
        max_age_hours=float(args.max_age_hours),
        require_gpu=bool(args.require_gpu),
    )
    out = write_ops_snapshot(project_dir=project_dir, payload=payload)
    print(str(out))
    return 0 if bool(payload.get("ok")) else 1


def cmd_ops_gate(args: argparse.Namespace) -> int:
    project_dir = _project_dir_from_cwd()
    policy_path = Path(args.policy) if args.policy else (project_dir / "conf" / "ops_policy.yaml")
    if not policy_path.is_absolute():
        policy_path = project_dir / policy_path
    policy = load_ops_policy(policy_path)

    payload = collect_ops_status(
        project_dir=project_dir,
        run_id=args.run_id,
        max_age_hours=float(policy.get("max_age_hours", 72.0)),
        require_gpu=bool(policy.get("require_gpu", False)),
    )
    gate = evaluate_ops_gate(payload, policy)
    if not bool(gate.get("ok")) and not bool(args.no_incident):
        incident = write_ops_incident(project_dir=project_dir, payload=gate)
        gate["incident_path"] = str(incident)
    print(json.dumps(gate, indent=2, ensure_ascii=True))
    return 0 if bool(gate.get("ok")) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gbdt_agent.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pf = sub.add_parser("preflight")
    p_pf.add_argument("--conf", required=True)
    p_pf.add_argument("--symbol", default="AAPL")
    p_pf.add_argument("--lookback-days", type=int, default=14)
    p_pf.set_defaults(func=cmd_preflight)

    p_run = sub.add_parser("run")
    p_run.add_argument("--conf", required=True)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--force-stage", default=None)
    p_run.add_argument("--force-unlock", action="store_true")
    p_run.add_argument("--stop-after-stage", default=None)
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report")
    p_rep.add_argument("--run-id", required=True)
    p_rep.add_argument("--conf", default=None)
    p_rep.set_defaults(func=cmd_report)

    p_tr = sub.add_parser("transition-report")
    p_tr.add_argument("--run-id", required=True)
    p_tr.add_argument("--target", default="colab")
    p_tr.set_defaults(func=cmd_transition_report)

    p_mig = sub.add_parser("migrate")
    mig_sub = p_mig.add_subparsers(dest="migrate_cmd", required=True)
    p_pack = mig_sub.add_parser("pack")
    p_pack.add_argument("--run-id", required=True)
    p_pack.add_argument("--out", required=True)
    p_pack.set_defaults(func=cmd_migrate_pack)
    p_res = mig_sub.add_parser("restore")
    p_res.add_argument("--archive", required=True)
    p_res.set_defaults(func=cmd_migrate_restore)

    p_colab = sub.add_parser("colab")
    colab_sub = p_colab.add_subparsers(dest="colab_cmd", required=True)
    p_cr = colab_sub.add_parser("restore")
    p_cr.add_argument("--drive-path", default="/content/drive/MyDrive/gbdt-stock-agent")
    p_cr.set_defaults(func=cmd_colab_restore)
    p_cs = colab_sub.add_parser("sync")
    p_cs.add_argument("--drive-path", default="/content/drive/MyDrive/gbdt-stock-agent")
    p_cs.set_defaults(func=cmd_colab_sync)

    p_ops = sub.add_parser("ops-status")
    p_ops.add_argument("--run-id", default=None)
    p_ops.add_argument("--max-age-hours", type=float, default=72.0)
    p_ops.add_argument("--require-gpu", action="store_true")
    p_ops.set_defaults(func=cmd_ops_status)

    p_ops_snap = sub.add_parser("ops-snapshot")
    p_ops_snap.add_argument("--run-id", default=None)
    p_ops_snap.add_argument("--max-age-hours", type=float, default=72.0)
    p_ops_snap.add_argument("--require-gpu", action="store_true")
    p_ops_snap.set_defaults(func=cmd_ops_snapshot)

    p_ops_gate = sub.add_parser("ops-gate")
    p_ops_gate.add_argument("--run-id", default=None)
    p_ops_gate.add_argument("--policy", default="conf/ops_policy.yaml")
    p_ops_gate.add_argument("--no-incident", action="store_true")
    p_ops_gate.set_defaults(func=cmd_ops_gate)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
