#!/bin/bash
# =============================================================================
# Download & Prepare All 35 DOMINO Tasks for PUMA Training
# =============================================================================
# This script:
#   1. Downloads aloha-agilex_clean_level1_50.zip for all 35 tasks from HuggingFace
#   2. Extracts them into the correct folder structure
#   3. Runs Step 1: RoboTwin raw HDF5 -> ALOHA HDF5 conversion
#   4. Runs Step 2: ALOHA HDF5 -> LeRobot format conversion
#   5. Copies modality.json to each task's meta folder
#
# Usage (run inside Docker container):
#   cd /workspace/PUMA
#   bash scripts/download_and_prepare_all_tasks.sh
#
# Options:
#   --skip-download    Skip download step (if already downloaded)
#   --skip-step1       Skip Step 1 conversion (if already done)
#   --skip-step2       Skip Step 2 conversion (if already done)
#   --tasks "t1 t2"    Only process specific tasks (space-separated)
# =============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUMA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Paths
DATA_DIR="${PUMA_ROOT}/data"
RAW_DATA_DIR="${DATA_DIR}/raw_downloads"       # Downloaded zip files
EXTRACTED_DIR="${DATA_DIR}/extracted"           # Extracted raw data (per task)
ROBOTWIN_DATA_DIR="${DATA_DIR}/robotwin_data"   # Organized for robotwin2hdf5.sh
OUTPUT_HDF5_DIR="${DATA_DIR}/output_hdf5"       # Step 1 output
LEROBOT_DIR="${DATA_DIR}/lerobot_data"          # Step 2 output (LeRobot format)

# HuggingFace dataset info
HF_REPO="H-EmbodVis/DOMINO"
HF_BRANCH="main"
SETTING="aloha-agilex_clean_level1"
ZIP_SUFFIX="aloha-agilex_clean_level1_50.zip"
EPISODE_COUNT=50

# All 35 tasks from robotwin_dynamic_task mixture
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

# ============================================================================
# Parse Arguments
# ============================================================================
SKIP_DOWNLOAD=false
SKIP_STEP1=false
SKIP_STEP2=false
SELECTED_TASKS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-step1)
            SKIP_STEP1=true
            shift
            ;;
        --skip-step2)
            SKIP_STEP2=true
            shift
            ;;
        --tasks)
            IFS=' ' read -ra SELECTED_TASKS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use selected tasks or all tasks
if [ ${#SELECTED_TASKS[@]} -gt 0 ]; then
    TASKS=("${SELECTED_TASKS[@]}")
else
    TASKS=("${ALL_TASKS[@]}")
fi

# ============================================================================
# Helper Functions
# ============================================================================
log_info() {
    echo ""
    echo "========================================"
    echo "[INFO] $1"
    echo "========================================"
}

log_task() {
    echo "  >> [$1/${#TASKS[@]}] $2"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: '$1' is not installed. Please install it first."
        exit 1
    fi
}

# ============================================================================
# Pre-flight Checks
# ============================================================================
check_command python3
check_command unzip

# Check if huggingface_hub is available (for download)
if [ "$SKIP_DOWNLOAD" = false ]; then
    if ! python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null; then
        echo "Installing huggingface_hub for downloading..."
        pip install -q huggingface_hub
    fi
fi

# Create directories
mkdir -p "${RAW_DATA_DIR}" "${EXTRACTED_DIR}" "${ROBOTWIN_DATA_DIR}" "${OUTPUT_HDF5_DIR}" "${LEROBOT_DIR}"

# ============================================================================
# Phase 1: Download
# ============================================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
    log_info "Phase 1: Downloading ${#TASKS[@]} tasks from HuggingFace"

    TASK_IDX=0
    for task in "${TASKS[@]}"; do
        TASK_IDX=$((TASK_IDX + 1))
        ZIP_FILE="${RAW_DATA_DIR}/${task}_${ZIP_SUFFIX}"

        if [ -f "${ZIP_FILE}" ]; then
            log_task "${TASK_IDX}" "${task} - already downloaded, skipping"
            continue
        fi

        log_task "${TASK_IDX}" "${task} - downloading..."

        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_REPO}',
    filename='dataset/${task}/${ZIP_SUFFIX}',
    repo_type='dataset',
    revision='${HF_BRANCH}',
    local_dir='${RAW_DATA_DIR}/_hf_cache',
    local_dir_use_symlinks=False,
)
" 2>&1

        # Move downloaded file to expected location
        DOWNLOADED="${RAW_DATA_DIR}/_hf_cache/dataset/${task}/${ZIP_SUFFIX}"
        if [ -f "${DOWNLOADED}" ]; then
            mv "${DOWNLOADED}" "${ZIP_FILE}"
            echo "    Saved to: ${ZIP_FILE}"
        else
            echo "    WARNING: Download may have failed for ${task}"
        fi
    done

    # Clean up HF cache structure
    rm -rf "${RAW_DATA_DIR}/_hf_cache"
    log_info "Phase 1 Complete: Downloads finished"
else
    log_info "Phase 1: Skipping download (--skip-download)"
fi

# ============================================================================
# Phase 2: Extract & Organize
# ============================================================================
log_info "Phase 2: Extracting and organizing data"

TASK_IDX=0
for task in "${TASKS[@]}"; do
    TASK_IDX=$((TASK_IDX + 1))
    ZIP_FILE="${RAW_DATA_DIR}/${task}_${ZIP_SUFFIX}"
    TASK_EXTRACT_DIR="${EXTRACTED_DIR}/${task}"
    TASK_ROBOTWIN_DIR="${ROBOTWIN_DATA_DIR}/${task}/${SETTING}"

    # Check if already organized
    if [ -d "${TASK_ROBOTWIN_DIR}/data" ] && [ "$(find "${TASK_ROBOTWIN_DIR}/data" -name '*.hdf5' | head -1)" ]; then
        log_task "${TASK_IDX}" "${task} - already extracted & organized, skipping"
        continue
    fi

    if [ ! -f "${ZIP_FILE}" ]; then
        echo "    WARNING: ${ZIP_FILE} not found, skipping ${task}"
        continue
    fi

    log_task "${TASK_IDX}" "${task} - extracting..."

    # Extract
    mkdir -p "${TASK_EXTRACT_DIR}"
    unzip -q -o "${ZIP_FILE}" -d "${TASK_EXTRACT_DIR}"

    # The zip might extract with different directory structures.
    # We need to find the 'data' and 'instructions' directories.
    # Try to detect the actual extracted structure.
    FOUND_DATA=""

    # Case 1: dataset/task/setting/data/
    if [ -d "${TASK_EXTRACT_DIR}/${SETTING}/data" ]; then
        FOUND_DATA="${TASK_EXTRACT_DIR}/${SETTING}"
    # Case 2: setting/data/  (no task prefix)
    elif [ -d "${TASK_EXTRACT_DIR}/data" ]; then
        FOUND_DATA="${TASK_EXTRACT_DIR}"
    # Case 3: Search recursively for the data directory
    else
        FOUND_DATA=$(find "${TASK_EXTRACT_DIR}" -type d -name "data" -exec dirname {} \; | head -1)
    fi

    if [ -z "${FOUND_DATA}" ] || [ ! -d "${FOUND_DATA}/data" ]; then
        echo "    WARNING: Could not find data directory for ${task} after extraction"
        echo "    Contents: $(ls ${TASK_EXTRACT_DIR})"
        continue
    fi

    # Organize into robotwin2hdf5.sh expected structure:
    #   ROBOTWIN_DATA_DIR/task_name/setting/data/episodeN.hdf5
    #   ROBOTWIN_DATA_DIR/task_name/setting/instructions/episodeN.json
    mkdir -p "${TASK_ROBOTWIN_DIR}"

    # Copy or move data
    if [ -d "${FOUND_DATA}/data" ]; then
        cp -rn "${FOUND_DATA}/data" "${TASK_ROBOTWIN_DIR}/" 2>/dev/null || true
    fi
    if [ -d "${FOUND_DATA}/instructions" ]; then
        cp -rn "${FOUND_DATA}/instructions" "${TASK_ROBOTWIN_DIR}/" 2>/dev/null || true
    fi
    if [ -f "${FOUND_DATA}/scene_info.json" ]; then
        cp -n "${FOUND_DATA}/scene_info.json" "${TASK_ROBOTWIN_DIR}/" 2>/dev/null || true
    fi
    if [ -f "${FOUND_DATA}/seed.txt" ]; then
        cp -n "${FOUND_DATA}/seed.txt" "${TASK_ROBOTWIN_DIR}/" 2>/dev/null || true
    fi

    # Verify
    EPISODE_FILES=$(find "${TASK_ROBOTWIN_DIR}/data" -name "*.hdf5" 2>/dev/null | wc -l)
    echo "    Organized: ${EPISODE_FILES} episodes in ${TASK_ROBOTWIN_DIR}"
done

log_info "Phase 2 Complete: All tasks extracted and organized"

# ============================================================================
# Phase 3: Step 1 - RoboTwin Raw -> ALOHA HDF5
# ============================================================================
if [ "$SKIP_STEP1" = false ]; then
    log_info "Phase 3: Step 1 - Converting RoboTwin raw data to ALOHA HDF5"

    export ROBOTWIN_DATA_PATH="${ROBOTWIN_DATA_DIR}"
    export DATA_PATH="${OUTPUT_HDF5_DIR}"

    TASK_IDX=0
    for task in "${TASKS[@]}"; do
        TASK_IDX=$((TASK_IDX + 1))
        ALOHA_OUTPUT="${OUTPUT_HDF5_DIR}/aloha_hdf5/${task}-${SETTING}-${EPISODE_COUNT}"

        # Check if already converted
        if [ -d "${ALOHA_OUTPUT}" ] && [ "$(find "${ALOHA_OUTPUT}" -type d -name 'episode_*' | head -1)" ]; then
            EXISTING=$(find "${ALOHA_OUTPUT}" -type d -name 'episode_*' | wc -l)
            log_task "${TASK_IDX}" "${task} - Step 1 already done (${EXISTING} episodes), skipping"
            continue
        fi

        # Check input exists
        if [ ! -d "${ROBOTWIN_DATA_DIR}/${task}/${SETTING}/data" ]; then
            echo "    WARNING: No input data for ${task}, skipping Step 1"
            continue
        fi

        log_task "${TASK_IDX}" "${task} - running Step 1..."

        bash "${SCRIPT_DIR}/robotwin2lerobot/robotwin2hdf5.sh" "${task}" "${SETTING}" "${EPISODE_COUNT}" || {
            echo "    WARNING: Step 1 failed for ${task}, continuing..."
            continue
        }

        echo "    Step 1 complete for ${task}"
    done

    log_info "Phase 3 Complete: Step 1 conversions finished"
else
    log_info "Phase 3: Skipping Step 1 (--skip-step1)"
fi

# ============================================================================
# Phase 4: Step 2 - ALOHA HDF5 -> LeRobot Format
# ============================================================================
if [ "$SKIP_STEP2" = false ]; then
    log_info "Phase 4: Step 2 - Converting ALOHA HDF5 to LeRobot format"

    export HF_LEROBOT_HOME="${LEROBOT_DIR}"

    TASK_IDX=0
    for task in "${TASKS[@]}"; do
        TASK_IDX=$((TASK_IDX + 1))
        ALOHA_INPUT="${OUTPUT_HDF5_DIR}/aloha_hdf5/${task}-${SETTING}-${EPISODE_COUNT}"
        LEROBOT_OUTPUT="${LEROBOT_DIR}/${task}"

        # Check if already converted
        if [ -d "${LEROBOT_OUTPUT}/meta" ] && [ -f "${LEROBOT_OUTPUT}/meta/info.json" ]; then
            log_task "${TASK_IDX}" "${task} - Step 2 already done, skipping"
            continue
        fi

        # Check input exists
        if [ ! -d "${ALOHA_INPUT}" ]; then
            echo "    WARNING: No ALOHA HDF5 data for ${task}, skipping Step 2"
            continue
        fi

        log_task "${TASK_IDX}" "${task} - running Step 2..."

        bash "${SCRIPT_DIR}/robotwin2lerobot/hdf52lerobot.sh" "${ALOHA_INPUT}" "${task}" || {
            echo "    WARNING: Step 2 failed for ${task}, continuing..."
            continue
        }

        echo "    Step 2 complete for ${task}"
    done

    log_info "Phase 4 Complete: Step 2 conversions finished"
else
    log_info "Phase 4: Skipping Step 2 (--skip-step2)"
fi

# ============================================================================
# Phase 5: Copy modality.json to all tasks
# ============================================================================
log_info "Phase 5: Copying modality.json to all task meta directories"

MODALITY_SRC="${PUMA_ROOT}/examples/Robotwin/train_files/modality.json"

if [ ! -f "${MODALITY_SRC}" ]; then
    echo "WARNING: modality.json not found at ${MODALITY_SRC}"
    echo "Please copy it manually to each task's meta directory."
else
    TASK_IDX=0
    for task in "${TASKS[@]}"; do
        TASK_IDX=$((TASK_IDX + 1))
        META_DIR="${LEROBOT_DIR}/${task}/meta"

        if [ -d "${META_DIR}" ]; then
            cp "${MODALITY_SRC}" "${META_DIR}/modality.json"
            log_task "${TASK_IDX}" "${task} - modality.json copied"
        else
            echo "    WARNING: ${META_DIR} does not exist, skipping"
        fi
    done
fi

log_info "Phase 5 Complete: modality.json copied"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           All Tasks Preparation Complete!                ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Tasks processed: ${#TASKS[@]}                                    ║"
echo "║                                                          ║"
echo "║  Data locations:                                         ║"
echo "║    Raw downloads : ${RAW_DATA_DIR}"
echo "║    RoboTwin data : ${ROBOTWIN_DATA_DIR}"
echo "║    ALOHA HDF5    : ${OUTPUT_HDF5_DIR}"
echo "║    LeRobot data  : ${LEROBOT_DIR}"
echo "║                                                          ║"
echo "║  Next step: Run training                                 ║"
echo "║    export DATA_ROOT_DIR=${LEROBOT_DIR}"
echo "║    bash scripts/run_scripts/run_lerobot_robotwin_puma.sh ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
