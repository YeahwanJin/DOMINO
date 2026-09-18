#!/bin/bash
# Usage: bash train.sh <repo_id> <exp_name> [n_gpus] [config] [ckpt]
#
# Examples:
#   DynamicVLA + world model:
#     bash train.sh all domino_all_tasks_with_wm 1 configs/domino_dynamicvla.yaml \
#         runs/pretrained/dynamic-vla-DOM
#   SmolVLA baseline (experiment 1):
#     bash train.sh all domino_smolvla 1 configs/domino_smolvla.yaml lerobot/smolvla_base
#   SmolVLA + world model (experiment 2):
#     bash train.sh all domino_smolvla_wm 1 configs/domino_smolvla_wm.yaml lerobot/smolvla_base
# Run inside the DynamicVLA conda environment.

set -e

repo_id=${1:-adjust_bottle}
exp_name=${2:-domino_adjust_bottle}
n_gpus=${3:-1}
config=${4:-configs/domino_dynamicvla.yaml}
ckpt=${5:-runs/pretrained/dynamic-vla-DOM}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"

# `ckpt` is either a local run directory or a Hugging Face repo id such as
# `lerobot/smolvla_base`; run.py resolves both. When it is neither, core.train falls
# back to the config's POLICY.CHECKPOINT (or trains from scratch if that is empty).
ckpt_arg=()
if [ -n "${ckpt}" ]; then
    ckpt_arg=(-p "${ckpt}")
fi

torchrun --nproc_per_node="${n_gpus}" run.py \
    -c "${config}" \
    -d "${repo_id}" \
    -e "${exp_name}" \
    "${ckpt_arg[@]}"
