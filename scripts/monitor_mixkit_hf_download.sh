#!/usr/bin/env bash
set -uo pipefail

REPO_ID="${REPO_ID:-LanguageBind/Open-Sora-Plan-v1.1.0}"
INCLUDE_GLOB="${INCLUDE_GLOB:-all_mixkit/**}"
LOCAL_DIR="${LOCAL_DIR:-/root/autodl-tmp/Physics_worldmodel/datasets/Open-Sora-Plan-v1.1.0}"
MAX_WORKERS="${MAX_WORKERS:-4}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
DOWNLOAD_TIMEOUT_SECONDS="${DOWNLOAD_TIMEOUT_SECONDS:-21600}"
CONDA_ROOT="${CONDA_ROOT:-/root/autodl-tmp/miniconda3}"
CONDA_ENV="${CONDA_ENV:-huggingface}"
LOG_FILE="${LOG_FILE:-${LOCAL_DIR}/mixkit_hf_watchdog.log}"

mkdir -p "${LOCAL_DIR}" "$(dirname "${LOG_FILE}")"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_FILE}"
}

download_running() {
    pgrep -af "huggingface-cli download ${REPO_ID}" | grep -v "monitor_mixkit_hf_download" >/dev/null && return 0
    pgrep -af "hf download ${REPO_ID}" | grep -v "monitor_mixkit_hf_download" >/dev/null && return 0
    return 1
}

run_download() {
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"

    export HF_ENDPOINT="https://hf-mirror.com"
    export HF_HUB_ENABLE_HF_TRANSFER=1

    log "Starting/resuming download with HF_ENDPOINT=${HF_ENDPOINT}, HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}, max_workers=${MAX_WORKERS}, timeout=${DOWNLOAD_TIMEOUT_SECONDS}s"
    python -c 'import hf_transfer; print("hf_transfer import ok")'

    timeout "${DOWNLOAD_TIMEOUT_SECONDS}" huggingface-cli download \
        "${REPO_ID}" \
        --repo-type dataset \
        --include "${INCLUDE_GLOB}" \
        --local-dir "${LOCAL_DIR}" \
        --max-workers "${MAX_WORKERS}"
}

log "Watchdog started for ${REPO_ID} include=${INCLUDE_GLOB}"
log "Local dir: ${LOCAL_DIR}"

while true; do
    if download_running; then
        size="$(du -sh "${LOCAL_DIR}" 2>/dev/null | awk '{print $1}')"
        log "Download process is active. local_size=${size:-unknown}. Sleeping ${SLEEP_SECONDS}s."
        sleep "${SLEEP_SECONDS}"
        continue
    fi

    log "No active Hugging Face download process found. Restarting/resuming now."
    run_download 2>&1 | tee -a "${LOG_FILE}"
    status=${PIPESTATUS[0]}

    if [[ "${status}" -eq 0 ]]; then
        size="$(du -sh "${LOCAL_DIR}" 2>/dev/null | awk '{print $1}')"
        log "Download command exited successfully. local_size=${size:-unknown}. Watchdog stopping."
        exit 0
    fi

    log "Download command failed with exit code ${status}. Sleeping ${SLEEP_SECONDS}s before retry."
    sleep "${SLEEP_SECONDS}"
done
