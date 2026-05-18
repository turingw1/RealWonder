#!/usr/bin/env bash
set -uo pipefail

WATCHDOG_SESSION="${WATCHDOG_SESSION:-dataset_watchdog}"
WATCHDOG_CMD="${WATCHDOG_CMD:-cd /root/autodl-tmp/Physics_worldmodel && bash RealWonder/scripts/monitor_mixkit_hf_download.sh}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
LOG_FILE="${LOG_FILE:-/root/autodl-tmp/Physics_worldmodel/datasets/Open-Sora-Plan-v1.1.0/mixkit_hf_watchdog_guard.log}"

mkdir -p "$(dirname "${LOG_FILE}")"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_FILE}"
}

log "Guard started for tmux session ${WATCHDOG_SESSION}"

while true; do
    if tmux has-session -t "${WATCHDOG_SESSION}" 2>/dev/null; then
        log "Session ${WATCHDOG_SESSION} is alive. Sleeping ${SLEEP_SECONDS}s."
    else
        log "Session ${WATCHDOG_SESSION} is missing. Starting it."
        tmux new-session -d -s "${WATCHDOG_SESSION}" "${WATCHDOG_CMD}"
    fi

    sleep "${SLEEP_SECONDS}"
done
