#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
REQ_FILE="${PROJECT_ROOT}/requirements.txt"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"

ROUNDS="${ROUNDS:-5}"
MIN_CLIENTS="${MIN_CLIENTS:-3}"
CLIENT_SAMPLES="${CLIENT_SAMPLES:-300}"
CLIENT_ROUNDS="${CLIENT_ROUNDS:-10}"
CLIENT_LR="${CLIENT_LR:-0.5}"

CLIENTS="${CLIENTS:-A B C}"

LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${PROJECT_ROOT}/.pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

EXPORT_DIR="${EXPORT_DIR:-${LOG_DIR}/exports}"
EXPORT_BASENAME="${EXPORT_BASENAME:-model_state_summary.json}"  # exact filename by default
AUTO_STOP="${AUTO_STOP:-1}"                                     # stop everything after export by default
SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-10}"                          # seconds to wait after controller exits
mkdir -p "${EXPORT_DIR}"

SERVER_PID_FILE="${PID_DIR}/server.pid"
CONTROLLER_PID_FILE="${PID_DIR}/controller.pid"
CLIENT_PIDS_FILE="${PID_DIR}/clients.pids"

PYTHON="${VENV_DIR}/bin/python"

ensure_venv() {
  if [[ ! -x "${PYTHON}" ]]; then
    echo "[*] Creating virtualenv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
  fi
  source "${VENV_DIR}/bin/activate"
  echo "[*] Installing requirements"
  pip install -r "${REQ_FILE}"
}

is_running() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] && ps -p "$(cat "$pidfile")" > /dev/null 2>&1
}

server_alive() {
  curl -fsS "${SERVER_URL}/status" > /dev/null 2>&1
}

wait_for_server() {
  echo -n "[*] Waiting for server ${SERVER_URL} "
  for _ in {1..60}; do
    if server_alive; then
      echo "OK"
      return 0
    fi
    echo -n "."
    sleep 1
  done
  echo "FAILED"
  echo "    → Tail logs with:  tail -n +1 -F ${LOG_DIR}/server.out ${LOG_DIR}/server.err"
  return 1
}

get_training_round() {
  local out rc
  for _ in {1..10}; do
    out="$(curl -fsS "${SERVER_URL}/model" 2>/dev/null | "${PYTHON}" - <<'PY'
import sys, json
try:
    d=json.load(sys.stdin)
    print(int(d.get("training_round", 0)))
except Exception:
    sys.exit(1)
PY
    )" && [[ "${out}" =~ ^[0-9]+$ ]] && { echo "${out}"; return 0; }
    sleep 0.5
  done
  return 1
}

wait_pid_exit() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  while kill -0 "$pid" >/dev/null 2>&1; do
    sleep 1
  done
}

start_server() {
  if is_running "${SERVER_PID_FILE}"; then
    echo "[i] Server already running (pid $(cat "${SERVER_PID_FILE}"))"
    return 0
  fi
  echo "[*] Starting server ..."
  ( cd "${PROJECT_ROOT}" && \
    nohup "${PYTHON}" -u -m server.server \
      > "${LOG_DIR}/server.out" 2> "${LOG_DIR}/server.err" & echo $! > "${SERVER_PID_FILE}" )
  wait_for_server
}

start_clients() {
  : > "${CLIENT_PIDS_FILE}"
  echo "[*] Starting clients: ${CLIENTS}"
  for cid in ${CLIENTS}; do
    ( cd "${PROJECT_ROOT}" && \
      nohup "${PYTHON}" -u -m client.client \
        --server "${SERVER_URL}" \
        --client-id "${cid}" \
        --samples "${CLIENT_SAMPLES}" \
        --rounds "${CLIENT_ROUNDS}" \
        --lr "${CLIENT_LR}" \
        > "${LOG_DIR}/client_${cid}.out" 2> "${LOG_DIR}/client_${cid}.err" & echo $! >> "${CLIENT_PIDS_FILE}" )
    sleep 0.2
  done
}

start_controller() {
  if is_running "${CONTROLLER_PID_FILE}"; then
    echo "[i] Controller already running (pid $(cat "${CONTROLLER_PID_FILE}"))"
    return 0
  fi
  echo "[*] Starting controller (rounds=${ROUNDS}, min_clients=${MIN_CLIENTS}) ..."
  ( cd "${PROJECT_ROOT}" && \
    nohup "${PYTHON}" -u -m controller.controller \
      --server "${SERVER_URL}" \
      --rounds "${ROUNDS}" \
      --min-clients "${MIN_CLIENTS}" \
      > "${LOG_DIR}/controller.out" 2> "${LOG_DIR}/controller.err" & echo $! > "${CONTROLLER_PID_FILE}" )
}

stop_pids_in_file() {
  local pidfile="$1"
  if [[ -f "$pidfile" ]]; then
    while read -r pid; do
      if [[ -n "${pid}" ]] && ps -p "${pid}" > /dev/null 2>&1; then
        kill "${pid}" 2>/dev/null || true
      fi
    done < "$pidfile"
    rm -f "$pidfile"
  fi
}

stop_all() {
  echo "[*] Stopping controller ..."
  stop_pids_in_file "${CONTROLLER_PID_FILE}"

  echo "[*] Stopping clients ..."
  stop_pids_in_file "${CLIENT_PIDS_FILE}"

  echo "[*] Stopping server ..."
  stop_pids_in_file "${SERVER_PID_FILE}"
}

status() {
  echo "---- STATUS ----"
  if server_alive; then
    echo "Server     : UP (HTTP 200)"
  else
    echo "Server     : DOWN (no HTTP response)"
  fi

  if [[ -f "${CLIENT_PIDS_FILE}" ]]; then
    idx=1
    while read -r pid; do
      if [[ -n "${pid}" ]] && ps -p "${pid}" > /dev/null 2>&1; then
        echo "Client[$idx]  : RUNNING (pid ${pid})"
      else
        echo "Client[$idx]  : STOPPED"
      fi
      idx=$((idx+1))
    done < "${CLIENT_PIDS_FILE}"
  else
    echo "Clients    : NONE"
  fi

  if is_running "${CONTROLLER_PID_FILE}"; then
    echo "Controller : RUNNING (pid $(cat "${CONTROLLER_PID_FILE}"))"
  else
    echo "Controller : STOPPED"
  fi

  echo "Export dir : ${EXPORT_DIR}"
  echo "Auto-stop  : ${AUTO_STOP}"
}

tail_logs() {
  echo "[*] Tailing logs (Ctrl+C to exit) ..."
  tail -n +1 -F \
    "${LOG_DIR}/server.out" \
    "${LOG_DIR}/server.err" \
    "${LOG_DIR}/controller.out" \
    "${LOG_DIR}/controller.err" \
    "${LOG_DIR}"/client_*.out \
    "${LOG_DIR}"/client_*.err 2>/dev/null || true
}

wait_and_export_after_controller() {
  if [[ ! -f "${CONTROLLER_PID_FILE}" ]]; then
    echo "[!] No controller PID file; cannot wait/export"
    return 1
  fi
  local ctrl_pid
  ctrl_pid="$(cat "${CONTROLLER_PID_FILE}")"

  echo "[*] Waiting for controller to complete (pid ${ctrl_pid})..."
  wait_pid_exit "${ctrl_pid}"
  echo "[*] Controller exited."

  if ! server_alive; then
    echo "[!] Server is unreachable after controller exit; attempting export anyway."
  fi

  local have_start="0" target_round=""
  if [[ -n "${START_ROUND:-}" && "${START_ROUND}" =~ ^[0-9]+$ && "${ROUNDS}" =~ ^[0-9]+$ ]]; then
    have_start="1"
    target_round=$(( START_ROUND + ROUNDS ))
    echo "[*] Waiting for server training_round to reach ${target_round} (timeout ${SETTLE_TIMEOUT}s) ..."
  else
    echo "[i] START_ROUND not available → stabilization wait (timeout ${SETTLE_TIMEOUT}s)"
  fi

  local deadline=$(( $(date +%s) + SETTLE_TIMEOUT ))
  local last="-1" same_count=0
  while true; do
    if ! server_alive; then
      echo -e "\n[!] Server down during settle wait. Continuing to export."
      break
    fi

    local cur
    cur="$(get_training_round || echo "")"
    if [[ -n "${cur}" && "${cur}" =~ ^[0-9]+$ ]]; then
      echo -ne "\r    current round=${cur}   "
      if [[ "${have_start}" == "1" ]]; then
        if (( cur >= target_round )); then echo; break; fi
      else
        if [[ "${cur}" == "${last}" ]]; then
          same_count=$((same_count+1))
        else
          same_count=0
        fi
        last="${cur}"
        if (( cur > 0 && same_count >= 2 )); then echo; break; fi
      fi
    fi

    if (( $(date +%s) > deadline )); then
      echo -e "\n[!] Timed out waiting for server to settle. Continuing to export."
      break
    fi
    sleep 1
  done

  local ts out
  ts="$(date +%Y%m%d_%H%M%S)"
  if [[ "${EXPORT_BASENAME}" == *.json ]]; then
    out="${EXPORT_DIR}/${EXPORT_BASENAME}"
  else
    out="${EXPORT_DIR}/${EXPORT_BASENAME}_${ts}.json"
  fi

  echo "[*] Exporting model to ${out} ..."
  if curl -fsS --retry 5 --retry-connrefused "${SERVER_URL}/export" -o "${out}"; then
    echo "[*] Export saved: ${out}"
  else
    echo "[!] Export FAILED (curl could not fetch ${SERVER_URL}/export)"
  fi

  if [[ "${AUTO_STOP}" = "1" ]]; then
    echo "[*] AUTO_STOP is enabled → stopping all processes"
    stop_all
  else
    echo "[i] AUTO_STOP=0 → leaving server/clients running"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  setup             Create venv and install requirements
  start             Start server, clients, and controller; export JSON; (AUTO_STOP=1 stops all)
  start-server      Start only the server
  start-clients     Start only clients
  start-controller  Start only controller
  stop              Stop all components
  status            Show process status
  logs              Tail all logs

Environment overrides:
  ROUNDS, MIN_CLIENTS, CLIENTS, CLIENT_SAMPLES, CLIENT_ROUNDS, CLIENT_LR
  SERVER_HOST, SERVER_PORT
  EXPORT_DIR, EXPORT_BASENAME (default: model_state_summary.json), AUTO_STOP (default: 1)
  SETTLE_TIMEOUT (default: 10s)
EOF
}

cmd="${1:-}"
case "${cmd}" in
  setup)
    ensure_venv
    ;;
  start)
    ensure_venv
    start_server
    start_clients

    if START_ROUND="$(get_training_round)"; then
      echo "[*] Starting server round is ${START_ROUND}"
    else
      echo "[i] Could not read starting training_round; proceeding with fallback wait"
      START_ROUND=""
    fi
    start_controller
    status
    wait_and_export_after_controller
    ;;
  start-server)
    ensure_venv
    start_server
    ;;
  start-clients)
    ensure_venv
    start_clients
    ;;
  start-controller)
    ensure_venv
    start_controller
    ;;
  stop)
    stop_all
    ;;
  status)
    status
    ;;
  logs)
    tail_logs
    ;;
  *)
    usage
    exit 1
    ;;
esac

