#!/usr/bin/env bash
# ==============================================================================
# Migrate eval_result/ from old layout to new layout
# ==============================================================================
# Old: eval_result/<task>/<policy>/<task_config>/<ckpt>/<timestamp>/
# New: eval_result/<policy>/<task_config>/<ckpt>/<shared_timestamp>/<task>/
#
# Strategy:
#   - For each task, pick the latest (most recent) timestamp directory
#   - Move its contents into the new shared-timestamp directory
#   - After moving, remove empty old directories
#   - Generate metrics_summary.txt and initial experimental_log.txt
#   - Remove old summary_*.txt files from eval_result/
# ==============================================================================

set -euo pipefail

EVAL_ROOT="eval_result"
SHARED_TIMESTAMP="2026-09-14 12:41:33"
POLICY="DynamicVLA"
TASK_CONFIG="demo_clean_dynamic"
CKPT="domino_dynamicvla"

NEW_RUN_DIR="${EVAL_ROOT}/${POLICY}/${TASK_CONFIG}/${CKPT}/${SHARED_TIMESTAMP}"

echo "============================================================"
echo "Migration: old eval_result layout -> new layout"
echo "  Shared timestamp : ${SHARED_TIMESTAMP}"
echo "  New run dir      : ${NEW_RUN_DIR}"
echo "============================================================"

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

mkdir -p "${NEW_RUN_DIR}"

for task in "${ALL_TASKS[@]}"; do
  OLD_BASE="${EVAL_ROOT}/${task}/${POLICY}/${TASK_CONFIG}/${CKPT}"

  if [[ ! -d "${OLD_BASE}" ]]; then
    echo "[SKIP] ${task}: no old data found"
    continue
  fi

  # Find the latest timestamp directory (alphabetically last = most recent)
  LATEST_TS=$(ls -1d "${OLD_BASE}/"*/ 2>/dev/null | sort | tail -n 1)

  if [[ -z "${LATEST_TS}" ]]; then
    echo "[SKIP] ${task}: no timestamp dirs found"
    continue
  fi

  LATEST_TS_NAME=$(basename "${LATEST_TS}")
  echo "[MOVE] ${task}: ${LATEST_TS_NAME} -> ${NEW_RUN_DIR}/${task}/"

  # Move latest result into new structure
  mkdir -p "${NEW_RUN_DIR}/${task}"
  # Use cp + rm instead of mv to handle cross-device safely
  cp -a "${LATEST_TS}/." "${NEW_RUN_DIR}/${task}/"
done

echo ""
echo "Files moved. Removing old task directories..."

for task in "${ALL_TASKS[@]}"; do
  OLD_TASK_DIR="${EVAL_ROOT}/${task}"
  if [[ -d "${OLD_TASK_DIR}" ]]; then
    rm -rf "${OLD_TASK_DIR}"
    echo "  Removed: ${OLD_TASK_DIR}"
  fi
done

# Remove old summary files
echo ""
echo "Removing old summary files..."
for f in "${EVAL_ROOT}"/summary_*.txt; do
  if [[ -f "$f" ]]; then
    rm "$f"
    echo "  Removed: $f"
  fi
done

# ==============================================================================
# Generate summary.txt
# ==============================================================================
SUMMARY_FILE="${NEW_RUN_DIR}/summary.txt"

echo "# Evaluation Summary" > "${SUMMARY_FILE}"
echo "# Timestamp: ${SHARED_TIMESTAMP}" >> "${SUMMARY_FILE}"
echo "# Policy: ${POLICY} | Task Config: ${TASK_CONFIG}" >> "${SUMMARY_FILE}"
echo "# Ckpt Setting: ${CKPT} | Test Num: 100 | Seed: 0" >> "${SUMMARY_FILE}"
echo "------------------------------------------------------------" >> "${SUMMARY_FILE}"

for task in "${ALL_TASKS[@]}"; do
  if [[ -f "${NEW_RUN_DIR}/${task}/_metrics.json" ]]; then
    echo "${task}: DONE" >> "${SUMMARY_FILE}"
  else
    echo "${task}: MISSING" >> "${SUMMARY_FILE}"
  fi
done

# ==============================================================================
# Generate metrics_summary.txt
# ==============================================================================
METRICS_SUMMARY="${NEW_RUN_DIR}/metrics_summary.txt"

echo "# Per-Task Metrics Summary" > "${METRICS_SUMMARY}"
echo "# Timestamp: ${SHARED_TIMESTAMP}" >> "${METRICS_SUMMARY}"
echo "# Policy: ${POLICY} | Task Config: ${TASK_CONFIG} | Ckpt: ${CKPT}" >> "${METRICS_SUMMARY}"
echo "# Test Num: 100 | Seed: 0" >> "${METRICS_SUMMARY}"
echo "============================================================" >> "${METRICS_SUMMARY}"
printf "%-35s %8s %10s %10s %10s %10s\n" \
  "Task" "SR(%)" "MS_mean" "MS_std" "RC_mean" "RC_std" >> "${METRICS_SUMMARY}"
echo "------------------------------------------------------------" >> "${METRICS_SUMMARY}"

_total_sr=0
_total_ms=0
_total_rc=0
_task_count=0

for task in "${ALL_TASKS[@]}"; do
  METRICS_FILE="${NEW_RUN_DIR}/${task}/_metrics.json"

  if [[ ! -f "${METRICS_FILE}" ]]; then
    printf "%-35s %8s %10s %10s %10s %10s\n" \
      "${task}" "N/A" "N/A" "N/A" "N/A" "N/A" >> "${METRICS_SUMMARY}"
    continue
  fi

  read -r SR MS_MEAN MS_STD RC_MEAN RC_STD <<< "$(python3 -c "
import json
with open('${METRICS_FILE}') as f:
    m = json.load(f)
def fmt(v):
    if v is None:
        return 'N/A'
    return f'{v:.2f}'
print(fmt(m.get('success_rate')),
      fmt(m.get('manipulation_score_mean')),
      fmt(m.get('manipulation_score_std')),
      fmt(m.get('route_completion_mean')),
      fmt(m.get('route_completion_std')))
")"

  printf "%-35s %8s %10s %10s %10s %10s\n" \
    "${task}" "${SR}" "${MS_MEAN}" "${MS_STD}" "${RC_MEAN}" "${RC_STD}" >> "${METRICS_SUMMARY}"

  if [[ "${SR}" != "N/A" ]]; then
    _total_sr=$(python3 -c "print(${_total_sr} + ${SR})")
    _total_ms=$(python3 -c "print(${_total_ms} + ${MS_MEAN})")
    _total_rc=$(python3 -c "print(${_total_rc} + ${RC_MEAN})")
    _task_count=$((_task_count + 1))
  fi
done

echo "------------------------------------------------------------" >> "${METRICS_SUMMARY}"
if [[ ${_task_count} -gt 0 ]]; then
  AVG_SR=$(python3 -c "print(f'{${_total_sr} / ${_task_count}:.2f}')")
  AVG_MS=$(python3 -c "print(f'{${_total_ms} / ${_task_count}:.2f}')")
  AVG_RC=$(python3 -c "print(f'{${_total_rc} / ${_task_count}:.2f}')")
  printf "%-35s %8s %10s %10s %10s %10s\n" \
    "AVERAGE (${_task_count} tasks)" "${AVG_SR}" "${AVG_MS}" "-" "${AVG_RC}" "-" >> "${METRICS_SUMMARY}"
fi
echo "============================================================" >> "${METRICS_SUMMARY}"

# ==============================================================================
# Create initial experimental_log.txt
# ==============================================================================
EXP_LOG="${EVAL_ROOT}/experimental_log.txt"

{
  echo "================================================================"
  echo "Run: ${SHARED_TIMESTAMP}"
  echo "Policy: ${POLICY} | Task Config: ${TASK_CONFIG} | Ckpt: ${CKPT}"
  echo "Test Num: 100 | Seed: 0 | Tasks: ${#ALL_TASKS[@]}"
  echo "Result Dir: ${NEW_RUN_DIR}"
  echo "----------------------------------------------------------------"
  tail -n +6 "${METRICS_SUMMARY}"
  echo ""
} >> "${EXP_LOG}"

echo ""
echo "============================================================"
echo "Migration complete!"
echo ""
echo "New structure:"
echo "  ${NEW_RUN_DIR}/"
echo "    ├── <task_name>/  (35 task directories)"
echo "    ├── summary.txt"
echo "    └── metrics_summary.txt"
echo "  ${EXP_LOG}"
echo "============================================================"
echo ""
cat "${METRICS_SUMMARY}"
