#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONF_PATH="${CONF_PATH:-conf/default.yaml}"
DRIVE_PATH="${DRIVE_PATH:-/content/drive/MyDrive/gbdt-stock-agent}"
MAX_AGE_HOURS="${MAX_AGE_HOURS:-72}"
OPS_POLICY="${OPS_POLICY:-conf/ops_policy.yaml}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python -m gbdt_agent.cli preflight --conf "${CONF_PATH}"
python -m gbdt_agent.cli run --conf "${CONF_PATH}" --resume
RUN_ID="$(python -c 'import json; print(json.load(open("state/last_run_state.json"))["run_id"])')"
python -m gbdt_agent.cli report --run-id "${RUN_ID}" --conf "${CONF_PATH}"
python -m gbdt_agent.cli transition-report --run-id "${RUN_ID}" --target colab
SNAPSHOT_RC=0
python -m gbdt_agent.cli ops-snapshot --run-id "${RUN_ID}" --max-age-hours "${MAX_AGE_HOURS}" --require-gpu || SNAPSHOT_RC=$?

GATE_RC=0
python -m gbdt_agent.cli ops-gate --run-id "${RUN_ID}" --policy "${OPS_POLICY}" || GATE_RC=$?
python -m gbdt_agent.cli colab sync --drive-path "${DRIVE_PATH}"

FINAL_RC=0
if [[ "${SNAPSHOT_RC}" -ne 0 ]]; then
  FINAL_RC="${SNAPSHOT_RC}"
fi
if [[ "${GATE_RC}" -ne 0 ]]; then
  FINAL_RC="${GATE_RC}"
fi

echo "ops_autopilot_done run_id=${RUN_ID} snapshot_rc=${SNAPSHOT_RC} gate_rc=${GATE_RC}"
exit "${FINAL_RC}"
