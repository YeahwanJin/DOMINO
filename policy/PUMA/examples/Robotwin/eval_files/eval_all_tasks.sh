#!/bin/bash
set -e

# ==============================================================================
# Batch Evaluation Script for all 35 RoboTwin / DOMINO tasks
# ==============================================================================

TEST_NUM=${1:-10}                  # Default: 10 episodes per task
TASK_CONFIG=${2:-demo_clean_dynamic} # Default: demo_clean_dynamic
CKPT_SETTING=${3:-puma_demo}
SEED=${4:-0}
GPU_ID=${5:-0}
PORT=${6:-9001}
HOST=${7:-172.17.0.2}

ALL_35_TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    "click_alarmclock"
    "click_bell"
    "dump_bin_bigbin"
    "grab_roller"
    "handover_block"
    "handover_mic"
    "hanging_mug"
    "move_can_pot"
    "move_pillbottle_pad"
    "move_playingcard_away"
    "move_stapler_pad"
    "place_a2b_left"
    "place_a2b_right"
    "place_bread_basket"
    "place_bread_skillet"
    "place_can_basket"
    "place_container_plate"
    "place_empty_cup"
    "place_fan"
    "place_mouse_pad"
    "place_object_basket"
    "place_object_scale"
    "place_object_stand"
    "place_phone_stand"
    "place_shoe"
    "press_stapler"
    "put_bottles_dustbin"
    "put_object_cabinet"
    "rotate_qrcode"
    "scan_object"
    "shake_bottle"
    "shake_bottle_horizontally"
    "stamp_seal"
)

TOTAL_TASKS=${#ALL_35_TASKS[@]}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/workspace/DOMINO/eval_logs/batch_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "======================================================================"
echo "🚀 Starting RoboTwin Batch Evaluation (${TOTAL_TASKS} tasks)"
echo "   - Episodes per task: ${TEST_NUM}"
echo "   - Task Config:       ${TASK_CONFIG}"
echo "   - Policy Checkpoint: ${CKPT_SETTING}"
echo "   - Server Target:     ws://${HOST}:${PORT}"
echo "   - Log Directory:     ${LOG_DIR}"
echo "======================================================================"

SUMMARY_FILE="${LOG_DIR}/summary.txt"
echo "RoboTwin Evaluation Summary - ${TIMESTAMP}" > "${SUMMARY_FILE}"
echo "Config: ${TASK_CONFIG} | Episodes/Task: ${TEST_NUM}" >> "${SUMMARY_FILE}"
echo "----------------------------------------------------------------------" >> "${SUMMARY_FILE}"
printf "%-30s | %-12s | %-12s\n" "Task Name" "Success Rate" "Status" >> "${SUMMARY_FILE}"
echo "----------------------------------------------------------------------" >> "${SUMMARY_FILE}"

CURRENT_IDX=0
START_TIME_ALL=$(date +%s)

for TASK_NAME in "${ALL_35_TASKS[@]}"; do
    CURRENT_IDX=$((CURRENT_IDX + 1))
    echo ""
    echo "======================================================================"
    echo "▶ [${CURRENT_IDX}/${TOTAL_TASKS}] Evaluating: ${TASK_NAME} (${TEST_NUM} eps)"
    echo "======================================================================"
    
    TASK_LOG="${LOG_DIR}/${TASK_NAME}.log"
    START_TIME_TASK=$(date +%s)

    # Run evaluation for current task
    if bash eval.sh "${TASK_NAME}" "${TASK_CONFIG}" "${CKPT_SETTING}" "${SEED}" "${GPU_ID}" "${PORT}" "${HOST}" "${TEST_NUM}" 2>&1 | tee "${TASK_LOG}"; then
        STATUS="OK"
    else
        STATUS="FAILED"
    fi

    END_TIME_TASK=$(date +%s)
    ELAPSED_TASK=$((END_TIME_TASK - START_TIME_TASK))

    # Parse success rate from result folder if available
    LATEST_RESULT=$(find /workspace/DOMINO/eval_result/${TASK_NAME}/model2robotwin_interface/${TASK_CONFIG}/${CKPT_SETTING}/ -name "_metrics.json" 2>/dev/null | sort | tail -n 1 || true)
    
    SR="N/A"
    if [ -n "${LATEST_RESULT}" ] && [ -f "${LATEST_RESULT}" ]; then
        SR=$(python3 -c "import json; data=json.load(open('${LATEST_RESULT}')); print(f\"{data.get('success_rate', 0)*100:.1f}%\")" 2>/dev/null || echo "N/A")
    fi

    printf "%-30s | %-12s | %-12s (%ds)\n" "${TASK_NAME}" "${SR}" "${STATUS}" "${ELAPSED_TASK}" | tee -a "${SUMMARY_FILE}"
done

END_TIME_ALL=$(date +%s)
TOTAL_ELAPSED=$((END_TIME_ALL - START_TIME_ALL))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))

echo ""
echo "======================================================================"
echo "🎉 All tasks finished in ${TOTAL_MIN} minutes!"
echo "📄 Summary saved to: ${SUMMARY_FILE}"
echo "======================================================================"
cat "${SUMMARY_FILE}"
