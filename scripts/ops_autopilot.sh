#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONF_PATH="${CONF_PATH:-conf/default.yaml}"
DRIVE_PATH="${DRIVE_PATH:-/content/drive/MyDrive/gbdt-stock-agent}"
MAX_AGE_HOURS="${MAX_AGE_HOURS:-72}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python -m gbdt_agent.cli preflight --conf "${CONF_PATH}"
RUN_ID="$(python -m gbdt_agent.cli run --conf "${CONF_PATH}" --resume | python -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')"
python -m gbdt_agent.cli report --run-id "${RUN_ID}" --conf "${CONF_PATH}"
python -m gbdt_agent.cli transition-report --run-id "${RUN_ID}" --target colab
python -m gbdt_agent.cli ops-snapshot --run-id "${RUN_ID}" --max-age-hours "${MAX_AGE_HOURS}" --require-gpu
python -m gbdt_agent.cli colab sync --drive-path "${DRIVE_PATH}"

echo "ops_autopilot_done run_id=${RUN_ID}"
