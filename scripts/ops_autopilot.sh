#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONF_PATH="${CONF_PATH:-conf/default.yaml}"
DRIVE_PATH="${DRIVE_PATH:-/content/drive/MyDrive/gbdt-stock-agent}"
SYNC_MODE="${SYNC_MODE:-quick}"
MAX_AGE_HOURS="${MAX_AGE_HOURS:-72}"
OPS_POLICY="${OPS_POLICY:-conf/ops_policy.yaml}"
RUN_MODEL_STABILITY="${RUN_MODEL_STABILITY:-0}"
RUN_FEATURE_STABILITY="${RUN_FEATURE_STABILITY:-0}"
RUN_READINESS_CHECK="${RUN_READINESS_CHECK:-0}"
READINESS_MODEL_RESULTS="${READINESS_MODEL_RESULTS:-reports/model_stability_prod_results.json}"
READINESS_FEATURE_RESULTS="${READINESS_FEATURE_RESULTS:-reports/feature_stability_prod_results.json}"
READINESS_REQUIRED_PERIODS="${READINESS_REQUIRED_PERIODS:-3}"
READINESS_MAX_STABILITY_AGE_HOURS="${READINESS_MAX_STABILITY_AGE_HOURS:-168}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if [[ "${RUN_MODEL_STABILITY}" == "1" ]]; then
  python scripts/model_stability_prod.py --base-conf "${CONF_PATH}" --promote-default
fi

if [[ "${RUN_FEATURE_STABILITY}" == "1" ]]; then
  python scripts/feature_stability_prod.py --base-conf "${CONF_PATH}" --promote-default
fi

python -m gbdt_agent.cli preflight --conf "${CONF_PATH}"
python -m gbdt_agent.cli run --conf "${CONF_PATH}" --resume
RUN_ID="$(python -c 'import json; print(json.load(open("state/last_run_state.json"))["run_id"])')"
python -m gbdt_agent.cli report --run-id "${RUN_ID}" --conf "${CONF_PATH}"
python -m gbdt_agent.cli transition-report --run-id "${RUN_ID}" --target colab
SNAPSHOT_RC=0
python -m gbdt_agent.cli ops-snapshot --run-id "${RUN_ID}" --max-age-hours "${MAX_AGE_HOURS}" --require-gpu || SNAPSHOT_RC=$?

GATE_RC=0
python -m gbdt_agent.cli ops-gate --run-id "${RUN_ID}" --policy "${OPS_POLICY}" || GATE_RC=$?
python -m gbdt_agent.cli colab sync --drive-path "${DRIVE_PATH}" --mode "${SYNC_MODE}" --run-id "${RUN_ID}"

READINESS_RC=0
if [[ "${RUN_READINESS_CHECK}" == "1" ]]; then
  python scripts/prod_readiness_check.py \
    --project-dir "${ROOT_DIR}" \
    --policy "${OPS_POLICY}" \
    --model-results "${READINESS_MODEL_RESULTS}" \
    --feature-results "${READINESS_FEATURE_RESULTS}" \
    --required-periods "${READINESS_REQUIRED_PERIODS}" \
    --max-stability-age-hours "${READINESS_MAX_STABILITY_AGE_HOURS}" \
    --strict || READINESS_RC=$?
fi

FINAL_RC=0
if [[ "${SNAPSHOT_RC}" -ne 0 ]]; then
  FINAL_RC="${SNAPSHOT_RC}"
fi
if [[ "${GATE_RC}" -ne 0 ]]; then
  FINAL_RC="${GATE_RC}"
fi
if [[ "${READINESS_RC}" -ne 0 ]]; then
  FINAL_RC="${READINESS_RC}"
fi

echo "ops_autopilot_done run_id=${RUN_ID} snapshot_rc=${SNAPSHOT_RC} gate_rc=${GATE_RC} readiness_rc=${READINESS_RC} model_stability=${RUN_MODEL_STABILITY} feature_stability=${RUN_FEATURE_STABILITY} readiness_check=${RUN_READINESS_CHECK}"
exit "${FINAL_RC}"
