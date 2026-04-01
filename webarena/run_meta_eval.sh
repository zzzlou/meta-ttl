#!/bin/bash

set -e

ACTOR="google/gemini-3-flash-preview"
MAX_EPISODES=4
MAX_CONCURRENT=4

ID_TASKS="1,3,7,19,20,28,29,38,42,43,44,46,49,52,53,55,62,69,78,81,89,91,96,99,103,104,112,118,122,156"
OOD_TASKS="0,2,5,8,13,14,17,21,26,27,30,32,33,40,47,48,50,51,59,70,80,88,97,98,101,106,107,108,110,113,114,115,116,125,126,131,132,133,134,135,136,137,138,139,140,141,145,149,150,151,152,153,154,160"

LOG_BASE="logs/meta_eval"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_eval() {
    local META_MODEL="$1"
    local PROMPT_NAME="$2"
    local SPLIT="$3"
    local TASKS="$4"
    local TAG="$5"

    local LOG_DIR="${LOG_BASE}/${TAG}_${SPLIT}_${TIMESTAMP}"

    echo ""
    echo "=================================================================="
    echo "  META: ${META_MODEL}"
    echo "  ACTOR: ${ACTOR}"
    echo "  PROMPT: ${PROMPT_NAME}"
    echo "  SPLIT: ${SPLIT}"
    echo "  LOG: ${LOG_DIR}"
    echo "  START: $(date)"
    echo "=================================================================="

    python -m webarena.evaluate_webarena \
        --meta_model "${META_MODEL}" \
        --actor_model "${ACTOR}" \
        --prompt_name "${PROMPT_NAME}" \
        --tasks "${TASKS}" \
        --max_episodes ${MAX_EPISODES} \
        --max_concurrent ${MAX_CONCURRENT} \
        --log_dir "${LOG_DIR}"

    echo "  FINISHED: $(date)"
    echo "=================================================================="
}

echo "Starting meta-agent evaluation sweep at $(date)"
echo "Timestamp: ${TIMESTAMP}"

run_eval "openai/gpt-5" "DEFAULT" "ID"  "${ID_TASKS}"  "gpt5_default"
run_eval "openai/gpt-5" "DEFAULT" "OOD" "${OOD_TASKS}" "gpt5_default"
run_eval "openai/gpt-5" "OPT_3"   "ID"  "${ID_TASKS}"  "gpt5_opt"
run_eval "openai/gpt-5" "OPT_3"   "OOD" "${OOD_TASKS}" "gpt5_opt"

run_eval "z-ai/glm-5" "DEFAULT" "ID"  "${ID_TASKS}"  "glm5_default"
run_eval "z-ai/glm-5" "DEFAULT" "OOD" "${OOD_TASKS}" "glm5_default"
run_eval "z-ai/glm-5" "OPT_2"   "ID"  "${ID_TASKS}"  "glm5_opt"
run_eval "z-ai/glm-5" "OPT_2"   "OOD" "${OOD_TASKS}" "glm5_opt"

run_eval "google/gemini-3-flash-preview" "DEFAULT" "ID"  "${ID_TASKS}"  "gemini3flash_default"
run_eval "google/gemini-3-flash-preview" "DEFAULT" "OOD" "${OOD_TASKS}" "gemini3flash_default"
run_eval "google/gemini-3-flash-preview" "OPT_4"   "ID"  "${ID_TASKS}"  "gemini3flash_opt"
run_eval "google/gemini-3-flash-preview" "OPT_4"   "OOD" "${OOD_TASKS}" "gemini3flash_opt"

echo ""
echo "All evaluations complete at $(date)"
echo "Results in: ${LOG_BASE}/*_${TIMESTAMP}"
