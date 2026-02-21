from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


class SecretMaskFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str]):
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not self._secrets:
            return True
        try:
            msg = record.getMessage()
        except Exception:
            return True

        masked = msg
        for s in self._secrets:
            if s and s in masked:
                masked = masked.replace(s, "****")

        if masked != msg:
            record.msg = masked
            record.args = ()
        return True


def mask_text(text: str, secrets: Iterable[str]) -> str:
    masked = str(text)
    for s in secrets:
        if s:
            masked = masked.replace(s, "****")
    return masked


def setup_logging(log_path: Path, level: str = "INFO", secrets: Optional[Iterable[str]] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"

    handlers: List[logging.Handler] = [
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ]
    if secrets:
        f = SecretMaskFilter(secrets)
        for h in handlers:
            h.addFilter(f)

    logging.basicConfig(level=lvl, format=fmt, handlers=handlers)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_capture(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@dataclass(frozen=True)
class EnvSnapshot:
    env_json_path: str
    pip_freeze_path: str
    gpu_info_path: str


def capture_env_snapshot(run_dir: Path) -> EnvSnapshot:
    run_dir.mkdir(parents=True, exist_ok=True)

    env_info = {
        "captured_at": utc_now_iso(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    env_json_path = run_dir / "env.json"
    env_json_path.write_text(json.dumps(env_info, indent=2, ensure_ascii=True))

    pip_freeze_path = run_dir / "pip_freeze.txt"
    pip_freeze_path.write_text(_run_capture([sys.executable, "-m", "pip", "freeze"]))

    gpu_info_path = run_dir / "gpu_info.txt"
    gpu_info_path.write_text(_run_capture(["nvidia-smi"]))

    return EnvSnapshot(
        env_json_path=str(env_json_path),
        pip_freeze_path=str(pip_freeze_path),
        gpu_info_path=str(gpu_info_path),
    )


def get_default_secrets_to_mask() -> List[str]:
    secrets: List[str] = []
    key = os.environ.get("FMP_API_KEY")
    if key:
        secrets.append(key)
    return secrets
