#!/bin/bash
# Usage: bash train.sh <repo_id> <exp_name> [n_gpus] [config]
# Run inside the DynamicVLA conda environment.

set -e

repo_id=${1:-adjust_bottle}
exp_name=${2:-domino_adjust_bottle}
n_gpus=${3:-1}
config=${4:-configs/domino_dynamicvla.yaml}
ckpt=${5:-runs/pretrained/dynamic-vla-DOM}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"

ckpt_arg=()
if [ -n "${ckpt}" ] && [ -d "${ckpt}" ]; then
    ckpt_arg=(-p "${ckpt}")
fi

torchrun --nproc_per_node="${n_gpus}" run.py \
    -c "${config}" \
    -d "${repo_id}" \
    -e "${exp_name}" \
    "${ckpt_arg[@]}"
