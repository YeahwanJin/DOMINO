#!/bin/bash
# Build a LeRobot dataset from DOMINO episodes and attach Depth Anything V2 labels.
#
# Usage: bash process_data.sh <task_name> <task_config> [repo_id] [depth_model] [camera]
# Run inside the DynamicVLA conda environment.
#
# `camera` must match the head-camera feature name in the resulting LeRobot dataset
# AND the WORLD_MODEL_DEPTH_KEY of the world-model configs
# (configs/domino_dynamicvla.yaml, configs/domino_smolvla_wm.yaml), since the depth
# sidecars are looked up at <dataset_root>/depth/<camera>/episode_XXXXXX.npy.

set -e

task_name=${1}
task_config=${2}
repo_id=${3:-domino/${task_name}}
depth_model=${4:-depth-anything/Depth-Anything-V2-Small-hf}
camera=${5:-cam_high}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
raw_dir=${script_dir}/../../data/${task_name}/${task_config}

echo -e "\033[32m[1/2] Converting ${raw_dir} -> LeRobot dataset ${repo_id}\033[0m"
python "${script_dir}/scripts/robotwin2lerobot/convert_robotwin_to_lerobot.py" \
    --raw-dir "${raw_dir}" \
    --repo-id "${repo_id}" \
    --overwrite

echo -e "\033[32m[2/2] Generating depth labels with ${depth_model}\033[0m"
# --out-size must stay equal to WORLD_MODEL_GRID_SIZE * 8 (16 * 8 = 128).
python "${script_dir}/scripts/generate_depth.py" \
    --repo-id "${repo_id}" \
    --model "${depth_model}" \
    --camera "${camera}" \
    --out-size 128

echo -e "\033[32mDone. Train with: bash train.sh ${repo_id} <exp_name> <gpus>\033[0m"
