#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
MODE="${MODE:-hybrid}" # hybrid | glm_only
WINDOW_NAME="${WINDOW_NAME:-claude}"
PROXY_DIR="${PROXY_DIR:-$HOME/claude-router-proxy}"
HYBRID_PROXY_SCRIPT="${HYBRID_PROXY_SCRIPT:-${WORK_DIR}/scripts/hybrid_proxy.mjs}"
GLM_PROXY_SCRIPT="${GLM_PROXY_SCRIPT:-${WORK_DIR}/scripts/glm_only_proxy.mjs}"
ATTACH_TMUX="${ATTACH_TMUX:-1}"
FORCE_RELAUNCH_CLAUDE="${FORCE_RELAUNCH_CLAUDE:-0}"

case "${MODE}" in
  hybrid)
    SESSION_NAME="${SESSION_NAME:-cc-team}"
    PROXY_URL="${PROXY_URL:-http://localhost:8787}"
    LEADER_MODEL="${LEADER_MODEL:-claude-opus-4-6}"
    MEMBER_MODEL="${MEMBER_MODEL:-glm-4.7}"
    ;;
  glm_only)
    SESSION_NAME="${SESSION_NAME:-cc-team-glm}"
    PROXY_URL="${PROXY_URL:-http://localhost:8788}"
    LEADER_MODEL="${LEADER_MODEL:-glm-4.7}"
    MEMBER_MODEL="${MEMBER_MODEL:-glm-4.7}"
    ;;
  *)
    echo "ERROR: unsupported MODE=${MODE} (expected: hybrid | glm_only)" >&2
    exit 1
    ;;
esac

PROXY_PORT="${PROXY_URL##*:}"
PROXY_PID_FILE="${PROXY_PID_FILE:-${WORK_DIR}/.claude/proxy_${PROXY_PORT}.pid}"
PROXY_LOG_FILE="${PROXY_LOG_FILE:-${WORK_DIR}/.claude/proxy_${PROXY_PORT}.log}"

SKILL_PATH="${WORK_DIR}/.claude/skills/codex-review/SKILL.md"
SKILL_RUNNER="${WORK_DIR}/scripts/codex_review.sh"

check_proxy() {
  lsof -nP -iTCP:"${PROXY_PORT}" -sTCP:LISTEN >/dev/null 2>&1 || return 1
  local health
  health="$(curl -fsS "${PROXY_URL}/health" 2>/dev/null || true)"
  [[ -n "${health}" ]] || return 1

  if [[ "${MODE}" == "hybrid" ]]; then
    [[ "${health}" == *'"mode":"hybrid"'* ]] || return 1
  elif [[ "${MODE}" == "glm_only" ]]; then
    [[ "${health}" == *'"mode":"glm_only"'* ]] || return 1
  fi
}

start_proxy_hybrid() {
  if [[ ! -f "${HYBRID_PROXY_SCRIPT}" ]]; then
    echo "ERROR: hybrid proxy script not found: ${HYBRID_PROXY_SCRIPT}" >&2
    exit 1
  fi

  stop_existing_proxy_on_port

  mkdir -p "$(dirname "${PROXY_PID_FILE}")"
  (
    cd "${WORK_DIR}"
    nohup env \
      PORT="${PROXY_PORT}" \
      HYBRID_LEADER_MODEL="${LEADER_MODEL}" \
      HYBRID_MEMBER_MODEL="${MEMBER_MODEL}" \
      ZAI_CONFIG_PATH="${PROXY_DIR}/.env" \
      node "${HYBRID_PROXY_SCRIPT}" > "${PROXY_LOG_FILE}" 2>&1 & echo $! > "${PROXY_PID_FILE}"
  )
  sleep 0.5
  curl -fsS "${PROXY_URL}/health" | grep -q '"mode":"hybrid"'
}

start_proxy_glm_only() {
  if [[ ! -f "${GLM_PROXY_SCRIPT}" ]]; then
    echo "ERROR: glm-only proxy script not found: ${GLM_PROXY_SCRIPT}" >&2
    exit 1
  fi

  stop_existing_proxy_on_port

  mkdir -p "$(dirname "${PROXY_PID_FILE}")"
  (
    cd "${WORK_DIR}"
    nohup env \
      PORT="${PROXY_PORT}" \
      ZAI_CONFIG_PATH="${PROXY_DIR}/.env" \
      node "${GLM_PROXY_SCRIPT}" > "${PROXY_LOG_FILE}" 2>&1 & echo $! > "${PROXY_PID_FILE}"
  )
  sleep 0.5
  curl -fsS "${PROXY_URL}/health" | grep -q '"mode":"glm_only"'
}

stop_existing_proxy_on_port() {
  local pids
  pids="$(lsof -tiTCP:"${PROXY_PORT}" -sTCP:LISTEN || true)"
  [[ -z "${pids}" ]] && return
  echo "[INFO] stopping existing listener on :${PROXY_PORT} (${pids//$'\n'/ })"
  # shellcheck disable=SC2086
  kill ${pids} >/dev/null 2>&1 || true
  sleep 0.5
}

ensure_proxy() {
  if check_proxy; then
    echo "[OK] proxy healthy: ${PROXY_URL}/health"
  else
    echo "[INFO] proxy not healthy, starting mode=${MODE}"
    if [[ "${MODE}" == "hybrid" ]]; then
      start_proxy_hybrid
    else
      start_proxy_glm_only
    fi
    echo "[OK] proxy started: ${PROXY_URL}/health"
  fi
}

ensure_tmux_session() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[OK] tmux session exists: ${SESSION_NAME}"
  else
    tmux new-session -d -s "${SESSION_NAME}" -n "${WINDOW_NAME}"
    echo "[OK] tmux session created: ${SESSION_NAME}"
  fi

  if ! tmux list-windows -t "${SESSION_NAME}" -F "#{window_name}" | grep -Fxq "${WINDOW_NAME}"; then
    tmux new-window -t "${SESSION_NAME}" -n "${WINDOW_NAME}"
  fi
}

launch_claude_if_needed() {
  local target="${SESSION_NAME}:${WINDOW_NAME}.0"
  local current_cmd
  current_cmd="$(tmux display-message -p -t "${target}" "#{pane_current_command}" 2>/dev/null || true)"

  if [[ "${current_cmd}" == "claude" ]]; then
    if [[ "${FORCE_RELAUNCH_CLAUDE}" == "1" ]]; then
      echo "[INFO] restarting existing claude in ${target}"
      tmux send-keys -t "${target}" C-c
      sleep 0.4
    else
      echo "[OK] claude already running in ${target}"
      return
    fi
  fi

  tmux send-keys -t "${target}" \
    "cd '${WORK_DIR}' && CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 ANTHROPIC_BASE_URL='${PROXY_URL}' ANTHROPIC_MODEL='${LEADER_MODEL}' ANTHROPIC_SMALL_FAST_MODEL='${MEMBER_MODEL}' claude --teammate-mode tmux" C-m
  echo "[OK] claude launched in ${target} (mode=${MODE}, leader=${LEADER_MODEL}, member=${MEMBER_MODEL})"
}

print_skill_hints() {
  if [[ -f "${SKILL_PATH}" ]]; then
    echo "[OK] Codex skill found: ${SKILL_PATH}"
  else
    echo "[WARN] Codex skill not found: ${SKILL_PATH}"
  fi

  if [[ -x "${SKILL_RUNNER}" ]]; then
    echo "[OK] Skill runner ready: ${SKILL_RUNNER}"
  else
    echo "[WARN] Skill runner is not executable: ${SKILL_RUNNER}"
  fi

  cat <<EOF
[INFO] mode=${MODE}
[INFO] proxy=${PROXY_URL}
[INFO] leader_model=${LEADER_MODEL}
[INFO] member_model=${MEMBER_MODEL}
[INFO] proxy_log=${PROXY_LOG_FILE}

Next:
1) tmux pane inside Claude: create Agent Team.
2) In Claude task/tool, call Codex teammate skill with:
   scripts/codex_review.sh "実装内容またはレビュー観点"
EOF
}

main() {
  ensure_proxy
  ensure_tmux_session
  launch_claude_if_needed
  print_skill_hints
  if [[ "${ATTACH_TMUX}" == "1" ]]; then
    echo "[INFO] attaching tmux session: ${SESSION_NAME}"
    tmux attach -t "${SESSION_NAME}"
  else
    echo "[INFO] ATTACH_TMUX=0, skip tmux attach"
  fi
}

main "$@"
