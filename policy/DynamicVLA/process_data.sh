#!/bin/bash
# Build a LeRobot dataset from DOMINO episodes and attach Depth Anything V2 labels.
#
# Usage: bash process_data.sh <task_name> <task_config> [repo_id] [depth_model]
# Run inside the DynamicVLA conda environment.

set -e

task_name=${1}
task_config=${2}
repo_id=${3:-domino/${task_name}}
depth_model=${4:-depth-anything/Depth-Anything-V2-Small-hf}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
raw_dir=${script_dir}/../../data/${task_name}/${task_config}

echo -e "\033[32m[1/2] Converting ${raw_dir} -> LeRobot dataset ${repo_id}\033[0m"
python "${script_dir}/scripts/robotwin2lerobot/convert_robotwin_to_lerobot.py" \
    --raw-dir "${raw_dir}" \
    --repo-id "${repo_id}" \
    --overwrite

echo -e "\033[32m[2/2] Generating depth labels with ${depth_model}\033[0m"
python "${script_dir}/scripts/generate_depth.py" \
    --repo-id "${repo_id}" \
    --model "${depth_model}" \
    --camera head_camera \
    --out-size 128

echo -e "\033[32mDone. Train with: bash train.sh ${repo_id} <exp_name> <gpus>\033[0m"
