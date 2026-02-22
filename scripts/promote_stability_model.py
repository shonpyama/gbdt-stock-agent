#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote selected stability model params to default config.")
    parser.add_argument(
        "--results",
        default="reports/model_stability_prod_results.json",
        help="Path to model_stability_prod_results.json",
    )
    parser.add_argument(
        "--default-conf",
        default="conf/default.yaml",
        help="Path to default.yaml to update",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print proposed update")
    args = parser.parse_args()

    results_path = Path(args.results)
    default_conf_path = Path(args.default_conf)

    payload = json.loads(results_path.read_text())
    selected = payload.get("selected") or {}
    name = selected.get("name")
    params = selected.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"selected.params missing or invalid in {results_path}")

    cfg = _load_yaml(default_conf_path)
    cfg.setdefault("models", {}).setdefault("gbdt", {})["params"] = params

    out = {
        "results": str(results_path),
        "default_conf": str(default_conf_path),
        "selected_name": name,
        "params": params,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        default_conf_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps(out, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
