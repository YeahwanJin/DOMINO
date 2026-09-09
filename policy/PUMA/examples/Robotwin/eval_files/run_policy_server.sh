#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
export puma_python=python3
#encoder,vlm ckpt가 다른데 어떻게 통합했지? policy 컨테이너 기준
your_ckpt=/path/to/output/20260831-puma-robotwin_dynamic_task-puma-robotwin-dynamic-35task 
gpu_id=0
port=9001
device=cuda   # cuda | npu (Ascend inference, see docs/ascend_inference.md)

CUDA_VISIBLE_DEVICES=$gpu_id ${puma_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --device ${device} \
    --use_bf16
