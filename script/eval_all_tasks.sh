#!/usr/bin/env bash
# ==============================================================================
# Evaluate All 35 DOMINO Dynamic Tasks
# ==============================================================================
# Usage (inside domino-eval container):
#   bash script/eval_all_tasks.sh [options]
#
# Options:
#   --port <int>           Policy server port (default: 5555)
#   --config <path>        Deploy policy YAML (default: policy/DynamicVLA/deploy_policy.yml)
#   --task_config <str>    Task config name (default: demo_clean_dynamic)
#   --ckpt_setting <str>   Checkpoint setting name (default: domino_dynamicvla)
#   --test_num <int>       Number of episodes per task (default: 100)
#   --seed <int>           Evaluation seed (default: 0)
#   --tasks "<t1> <t2>"    Space-separated custom task list (default: all 35 tasks)
#
# Result directory layout:
#   eval_result/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/
#     ├── <task_name>/          (per-task result files & videos)
#     ├── summary.txt           (DONE / FAILED per task)
#     └── metrics_summary.txt   (success rate, MS, RC per task + average)
#   eval_result/experimental_log.txt   (accumulated across runs)
# ==============================================================================

set -u

PORT=5555
CONFIG="policy/DynamicVLA/deploy_policy.yml"
TASK_CONFIG="demo_clean_dynamic"
CKPT_SETTING="domino_dynamicvla"
TEST_NUM=100
SEED=0
CUSTOM_TASKS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --task_config)
      TASK_CONFIG="$2"
      shift 2
      ;;
    --ckpt_setting)
      CKPT_SETTING="$2"
      shift 2
      ;;
    --test_num)
      TEST_NUM="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --tasks)
      CUSTOM_TASKS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Extract policy_name from the deploy config YAML
POLICY_NAME=$(python3 -c "
import yaml, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['policy_name'])
")

ALL_TASKS=(
  adjust_bottle
  beat_block_hammer
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_block
  handover_mic
  hanging_mug
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_bread_skillet
  place_can_basket
  place_container_plate
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle
  shake_bottle_horizontally
  stamp_seal
)

if [[ -n "${CUSTOM_TASKS}" ]]; then
  read -ra TASKS <<< "${CUSTOM_TASKS}"
else
  TASKS=("${ALL_TASKS[@]}")
fi

TOTAL_TASKS=${#TASKS[@]}

# Shared timestamp for this entire evaluation run
EVAL_TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
RUN_DIR="eval_result/${POLICY_NAME}/${TASK_CONFIG}/${CKPT_SETTING}/${EVAL_TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "============================================================"
echo "Starting evaluation across ${TOTAL_TASKS} tasks"
echo "  Policy Name   : ${POLICY_NAME}"
echo "  Policy Config : ${CONFIG}"
echo "  Server Port   : ${PORT}"
echo "  Task Config   : ${TASK_CONFIG}"
echo "  Ckpt Setting  : ${CKPT_SETTING}"
echo "  Test Num/Task : ${TEST_NUM}"
echo "  Seed          : ${SEED}"
echo "  Run Dir       : ${RUN_DIR}"
echo "============================================================"

SUMMARY_FILE="${RUN_DIR}/summary.txt"

echo "# Evaluation Summary" > "${SUMMARY_FILE}"
echo "# Timestamp: ${EVAL_TIMESTAMP}" >> "${SUMMARY_FILE}"
echo "# Policy: ${POLICY_NAME} | Config: ${CONFIG} | Task Config: ${TASK_CONFIG}" >> "${SUMMARY_FILE}"
echo "# Ckpt Setting: ${CKPT_SETTING} | Test Num: ${TEST_NUM} | Seed: ${SEED}" >> "${SUMMARY_FILE}"
echo "------------------------------------------------------------" >> "${SUMMARY_FILE}"

INDEX=0
for task in "${TASKS[@]}"; do
  INDEX=$((INDEX + 1))
  echo -e "\n\033[1;34m[Task ${INDEX}/${TOTAL_TASKS}] Evaluating: ${task}\033[0m"

  python script/eval_policy_client.py \
    --port "${PORT}" \
    --config "${CONFIG}" \
    --overrides \
      --task_name "${task}" \
      --task_config "${TASK_CONFIG}" \
      --ckpt_setting "${CKPT_SETTING}" \
      --seed "${SEED}" \
      --test_num "${TEST_NUM}" \
      --eval_timestamp "${EVAL_TIMESTAMP}"

  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo -e "\033[1;31m[Task ${INDEX}/${TOTAL_TASKS}] ${task} failed with exit code ${EXIT_CODE}\033[0m"
    echo "${task}: FAILED (code ${EXIT_CODE})" >> "${SUMMARY_FILE}"
  else
    echo "${task}: DONE" >> "${SUMMARY_FILE}"
  fi
done

echo "" >> "${SUMMARY_FILE}"
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> "${SUMMARY_FILE}"

echo -e "\n============================================================"
echo "Evaluation completed! Summary saved to: ${SUMMARY_FILE}"
echo "============================================================"

# ==============================================================================
# Collect per-task metrics into a combined metrics summary
# ==============================================================================

METRICS_SUMMARY="${RUN_DIR}/metrics_summary.txt"

echo "# Per-Task Metrics Summary" > "${METRICS_SUMMARY}"
echo "# Timestamp: ${EVAL_TIMESTAMP}" >> "${METRICS_SUMMARY}"
echo "# Policy: ${POLICY_NAME} | Task Config: ${TASK_CONFIG} | Ckpt: ${CKPT_SETTING}" >> "${METRICS_SUMMARY}"
echo "# Test Num: ${TEST_NUM} | Seed: ${SEED}" >> "${METRICS_SUMMARY}"
echo "============================================================" >> "${METRICS_SUMMARY}"
printf "%-35s %8s %10s %10s %10s %10s %6s %6s\n" \
  "Task" "SR(%)" "MS_mean" "MS_std" "RC_mean" "RC_std" "P_col" "P_oob" >> "${METRICS_SUMMARY}"
echo "-----------------------------------------------------------------------------------------------" >> "${METRICS_SUMMARY}"

# Accumulators for overall stats
_total_sr=0
_total_ms=0
_total_rc=0
_total_col=0
_total_oob=0
_task_count=0

for task in "${TASKS[@]}"; do
  METRICS_FILE="${RUN_DIR}/${task}/_metrics.json"

  if [[ ! -f "${METRICS_FILE}" ]]; then
    printf "%-35s %8s %10s %10s %10s %10s %6s %6s\n" \
      "${task}" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" >> "${METRICS_SUMMARY}"
    continue
  fi

  read -r SR MS_MEAN MS_STD RC_MEAN RC_STD P_COL P_OOB <<< "$(python3 -c "
import json
with open('${METRICS_FILE}') as f:
    m = json.load(f)
def fmt(v):
    if v is None:
        return 'N/A'
    return f'{v:.2f}'
def fmti(v):
    if v is None:
        return 'N/A'
    return str(int(v))
print(fmt(m.get('success_rate')),
      fmt(m.get('manipulation_score_mean')),
      fmt(m.get('manipulation_score_std')),
      fmt(m.get('route_completion_mean')),
      fmt(m.get('route_completion_std')),
      fmti(m.get('penalty_clutter_collision_total')),
      fmti(m.get('penalty_out_of_bounds_total')))
")"

  printf "%-35s %8s %10s %10s %10s %10s %6s %6s\n" \
    "${task}" "${SR}" "${MS_MEAN}" "${MS_STD}" "${RC_MEAN}" "${RC_STD}" "${P_COL}" "${P_OOB}" >> "${METRICS_SUMMARY}"

  if [[ "${SR}" != "N/A" ]]; then
    _total_sr=$(python3 -c "print(${_total_sr} + ${SR})")
    _total_ms=$(python3 -c "print(${_total_ms} + ${MS_MEAN})")
    _total_rc=$(python3 -c "print(${_total_rc} + ${RC_MEAN})")
    _total_col=$((_total_col + ${P_COL:-0}))
    _total_oob=$((_total_oob + ${P_OOB:-0}))
    _task_count=$((_task_count + 1))
  fi
done

echo "-----------------------------------------------------------------------------------------------" >> "${METRICS_SUMMARY}"
if [[ ${_task_count} -gt 0 ]]; then
  AVG_SR=$(python3 -c "print(f'{${_total_sr} / ${_task_count}:.2f}')")
  AVG_MS=$(python3 -c "print(f'{${_total_ms} / ${_task_count}:.2f}')")
  AVG_RC=$(python3 -c "print(f'{${_total_rc} / ${_task_count}:.2f}')")
  printf "%-35s %8s %10s %10s %10s %10s %6s %6s\n" \
    "AVERAGE (${_task_count} tasks)" "${AVG_SR}" "${AVG_MS}" "-" "${AVG_RC}" "-" "${_total_col}" "${_total_oob}" >> "${METRICS_SUMMARY}"
else
  echo "No valid task metrics found." >> "${METRICS_SUMMARY}"
fi
echo "============================================================" >> "${METRICS_SUMMARY}"

echo ""
echo "Metrics summary saved to: ${METRICS_SUMMARY}"
cat "${METRICS_SUMMARY}"

# ==============================================================================
# Append to persistent experimental log
# ==============================================================================

EXP_LOG="eval_result/experimental_log.txt"

{
  echo ""
  echo "================================================================"
  echo "Run: ${EVAL_TIMESTAMP}"
  echo "Policy: ${POLICY_NAME} | Task Config: ${TASK_CONFIG} | Ckpt: ${CKPT_SETTING}"
  echo "Test Num: ${TEST_NUM} | Seed: ${SEED} | Tasks: ${TOTAL_TASKS}"
  echo "Result Dir: ${RUN_DIR}"
  echo "----------------------------------------------------------------"
  # Re-print the metrics table (skip header lines, just the data)
  tail -n +6 "${METRICS_SUMMARY}"
  echo ""
} >> "${EXP_LOG}"

echo ""
echo "Appended to experimental log: ${EXP_LOG}"
