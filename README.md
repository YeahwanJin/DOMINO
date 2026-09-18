<h2 align="center"> Towards Generalizable Robotic Manipulation in Dynamic Environments </h2>

<div align="center">
    <a href="https://arxiv.org/abs/2603.15620"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=Arxiv"></a>
    <a href="https://h-embodvis.github.io/DOMINO/"><img src="https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome"></a>
    <a href="https://huggingface.co/datasets/h-embodvis/DOMINO"><img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg?logo=huggingface"></a>
    <a href="https://huggingface.co/H-EmbodVis/PUMA"><img src="https://img.shields.io/badge/HuggingFace-Model-green.svg?logo=huggingface"></a>
    <a href="https://www.modelscope.cn/datasets/H-EmbodVis/DOMINO"><img src="https://img.shields.io/badge/ModelScope-Dataset-37CED1.svg?logo=modelscope"></a>
    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square&logo=apache"></a>

<h5 align="center"><em>Heng Fang<sup>1</sup>, Shangru Li<sup>1</sup>, Shuhan Wang<sup>1</sup>, Xuanyang Xi<sup>2</sup>, <a href="https://dk-liang.github.io/">Dingkang Liang</a><sup>1,†</sup>, <a href="https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=en">Xiang Bai</a><sup>1</sup> </em></h5>
<sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> Huawei Technologies Co. Ltd, <sup>†</sup> Corresponding Author
</div>


## 🔍 Overview

Dynamic manipulation requires robots to continuously adapt to moving objects and unpredictable environmental changes. Existing Vision-Language-Action (VLA) models rely on static single-frame observations, failing to capture essential spatiotemporal dynamics. We introduce **DOMINO**, a comprehensive benchmark for this underexplored frontier, and **PUMA**, a predictive architecture that couples historical motion cues with future state anticipation to achieve highly reactive embodied intelligence.

<div  align="center">    
 <img src="./assets/static/intro.png" width = "90%"  align=center />
</div>

<details>
  <summary>Abstract
  </summary>

Vision-Language-Action (VLA) models excel in static manipulation but struggle in dynamic environments with moving targets. This performance gap primarily stems from a scarcity of dynamic manipulation datasets and the reliance of mainstream VLAs on single-frame observations, restricting their spatiotemporal reasoning capabilities. To address this, we introduce DOMINO, a large-scale dataset and benchmark for generalizable dynamic manipulation, featuring 35 tasks with hierarchical complexities, over 110K expert trajectories, and a multi-dimensional evaluation suite. Through comprehensive experiments, we systematically evaluate existing VLAs on dynamic tasks, explore effective training strategies for dynamic awareness, and validate the generalizability of dynamic data. Furthermore, we propose PUMA, a dynamics-aware VLA architecture. By integrating scene-centric historical optical flow and specialized world queries to implicitly forecast object-centric future states, PUMA couples history-aware perception with short-horizon prediction. Results demonstrate that PUMA achieves state-of-the-art performance, yielding a 6.3% absolute improvement in success rate over baselines. Moreover, we show that training on dynamic data fosters robust spatiotemporal representations that transfer to static tasks.
</details>


### 📰 News

**[2026/08/18]** 🚀 PUMA now supports **training on Huawei Ascend NPUs**: NVIDIA weights train as-is with DeepSpeed ZeRO-2, and the CUDA path stays untouched — see the [Ascend training guide](policy/PUMA/docs/ascend_training.md).

**[2026/07/31]** 🚀 PUMA now runs **inference on Huawei Ascend NPUs**: NVIDIA-trained checkpoints work directly on Atlas 910 with no weight conversion — see the [Ascend inference guide](policy/PUMA/docs/ascend_inference.md).

**[2026/06/18]** 🎉 DOMINO has been accepted to **ECCV 2026**!

**[2026/05/29]** 🙏 Special thanks to the [Qwen team](https://github.com/QwenLM) for using DOMINO in [Qwen-VLA](https://arxiv.org/abs/2605.30280) as a **dynamic manipulation OOD benchmark**! We welcome everyone to try DOMINO for evaluating VLA robustness.

**[2026/04/22]** 🔥 DOMINO now supports the [StarVLA](https://github.com/starVLA/starVLA) codebase! Evaluation code is available [here](https://github.com/starVLA/starVLA/tree/starVLA_dev/examples/DOMINO).

**[2026/03/30]** 🚀 We now release the PUMA training/evaluation code and the [PUMA checkpoint](https://huggingface.co/H-EmbodVis/PUMA).

**[2026/03/28]** 🔥 The DOMINO dataset is now available on [Hugging Face](https://huggingface.co/datasets/h-embodvis/DOMINO) and [ModelScope](https://www.modelscope.cn/datasets/H-EmbodVis/DOMINO).

**[2026/03/24]** 🚀 We release the DOMINO benchmark code, including setup, data collection, and policy evaluation instructions.

**[2026/03/17]** 🎉 We release the [paper](https://arxiv.org/abs/2603.15620), [project homepage](https://h-embodvis.github.io/DOMINO/), and visual demos.


### 🎥 Visual Demos

More visual demos can be found on our [project homepage](https://h-embodvis.github.io/DOMINO/).

<div align="center">
  <img src="assets/static/gif/1.gif" width="32%" />
  <img src="assets/static/gif/2.gif" width="32%" />
  <img src="assets/static/gif/3.gif" width="32%" />
</div>
<div align="center">
  <img src="assets/static/gif/4.gif" width="32%" />
  <img src="assets/static/gif/5.gif" width="32%" />
  <img src="assets/static/gif/6.gif" width="32%" />
</div>

### ✨ Key Idea

* Current VLA models struggle with dynamic manipulation tasks due to a scarcity of dynamic datasets and a reliance on single-frame observations.
* We introduce DOMINO, a large-scale benchmark for dynamic manipulation comprising 35 tasks and over 110K expert trajectories.
* We propose PUMA, a dynamics-aware VLA architecture that integrates historical optical flow and world queries to forecast future object states.
* Training on dynamic data fosters robust spatiotemporal representations, demonstrating enhanced generalization capabilities.


## 📅 TODO
* [x] Release the paper
* [x] Release DOMINO benchmark code
* [x] Release DOMINO dataset on [HuggingFace](https://huggingface.co/datasets/h-embodvis/DOMINO) and [ModelScope](https://www.modelscope.cn/datasets/H-EmbodVis/DOMINO)
* [x] Release PUMA training code and evaluation code
* [x] Release PUMA checkpoint on [HuggingFace](https://huggingface.co/H-EmbodVis/PUMA)
* [x] Support [StarVLA](https://github.com/starVLA/starVLA) codebase (evaluation code available [here](https://github.com/starVLA/starVLA/tree/starVLA_dev/examples/DOMINO))
* [x] Add real-world evaluation results
* [x] Support Huawei Ascend NPUs — PUMA inference ([guide](policy/PUMA/docs/ascend_inference.md))
* [x] Support Huawei Ascend NPUs — PUMA training ([guide](policy/PUMA/docs/ascend_training.md))


## 🛠️ Getting Started

This project is divided into two main components that operate in separate environments and communicate via WebSockets:
- **DOMINO**: The simulation environment and data generation pipeline.
- **PUMA**: The Vision-Language-Action policy framework.

You will need to set up both environments to run the full pipeline.

### 1. DOMINO (Simulation & Data Pipeline)

#### 1.0. System Requirements
- **OS**: Linux (Windows/MacOS have limited or no support)
- **Hardware**: NVIDIA GPU (RTX recommended for ray tracing)
- **Software**: Python 3.10, CUDA 12.1 (Recommended), NVIDIA Driver >= 520

*Note: If running inside a Docker container, you must include the graphics capability to avoid Vulkan-related segmentation faults:*
```bash
docker run ... -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

#### 1.1. Installation Steps

**Step 1: Install System Dependencies**
Ensure Vulkan and FFmpeg are installed on your system:
```bash
sudo apt update
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools ffmpeg
```
*(Verify installations by running `vulkaninfo` and `ffmpeg -version`)*

**Step 2: Create Conda Environment**
```bash
conda create -n domino python=3.10 -y
conda activate domino
```

**Step 3: Clone and Install**
```bash
git clone https://github.com/h-embodvis/DOMINO.git
cd DOMINO

# Install basic environments and CuRobo
bash script/_install.sh
```
*Troubleshooting: If you encounter a CuRobo config path issue, run `python script/update_embodiment_config_path.py`. A failed PyTorch3D installation won't affect core functionality unless you are using 3D data.*

**Step3-2: docker image build / container run**
```bash
# 각각 빌드
docker build --target eval       -t domino-eval       .
docker build --target puma       -t domino-puma       .
docker build --target dynamicvla -t domino-dynamicvla .

# eval-container
docker run --rm -it --name domino-eval \
  --gpus '"device=1"' \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --device /dev/dri \
  -v $(pwd)/assets:/workspace/DOMINO/assets \
  -v $(pwd)/data:/workspace/DOMINO/data \
  domino-eval /bin/bash

# puma
docker run -it --name domino-puma \
  --gpus all \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v $(pwd)/assets:/workspace/DOMINO/assets \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/eval_logs:/workspace/DOMINO/eval_logs \
  -v $(pwd)/eval_result:/workspace/DOMINO/eval_result \
  domino-eval /bin/bash

# dynamicVLA
docker run -it -d --name domino-dynamicvla \
  --gpus all \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v $(pwd)/assets:/workspace/DOMINO/assets \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/eval_logs:/workspace/DOMINO/eval_logs \
  -v $(pwd)/eval_result:/workspace/DOMINO/eval_result \
  domino-eval /bin/bash

'''

**Step 4: Download Assets**
Download the required assets (RoboTwin-OD, Texture Library, and Embodiments). If you hit rate limits, log in to Hugging Face first (`huggingface-cli login`).
```bash
bash script/_download_assets.sh
```

#### 1.2. Data Collection

We provide an automated pipeline for data collection. You can collect data by running:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh adjust_bottle demo_clean_dynamic 0
```

After collection, the data will be stored under `data/${task_name}/${task_config}` in **HDF5 format**. For the full data collection process and common issues, please refer to the [RoboTwin Data Collection Tutorial](https://robotwin-platform.github.io/doc/usage/collect-data.html).

**Dynamic Task Configurations**

To enable dynamic environments, we introduce four specific configurations in the task config files (e.g., `task_config/demo_clean_dynamic.yml` and `task_config/demo_random_dynamic.yml`):

<details>
<summary><b>Click to view Dynamic Configurations</b></summary>

- `use_dynamic` (bool): Whether to enable dynamic motion in the environment (e.g., moving objects).
- `dynamic_level` (int): The complexity level of the dynamic motion (1, 2, or 3). Higher levels introduce more challenging dynamic behaviors.
- `dynamic_coefficient` (float): A scaling factor (default: 0.1) that controls the speed of the dynamic movements.
- `check_render_success` (bool): Whether to verify rendering success during data collection, ensuring that dynamic interactions do not cause visual or physical glitches.

</details>

For all other detailed configurations (like domain randomization, cameras, and data types), we maintain the original RoboTwin 2.0 settings. You can find more information in the [RoboTwin Configurations Tutorial](https://robotwin-platform.github.io/doc/usage/configurations.html).

#### 1.3. Policy Evaluation

To evaluate a trained policy, use the following command. The `task_config` field refers to the evaluation environment configuration, while the `ckpt_setting` field refers to the training data configuration used during policy learning.

```bash
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}

# Example: Evaluate a policy trained on `demo_clean_dynamic` and tested on `demo_clean_dynamic`
# bash eval.sh adjust_bottle demo_clean_dynamic demo_clean_dynamic 50 0 0
```

<details>
<summary><b>Click to view Dynamic Adaptations in Evaluation</b></summary>

To better evaluate dynamic manipulation, we have introduced several modifications in `script/eval_policy.py` and `script/eval_metrics.py`:

- **Enhanced Evaluation Metrics**: Alongside the standard Success Rate (SR), we introduce the **Manipulation Score (MS)**, a comprehensive metric that evaluates route completion while applying penalties for undesirable behaviors (e.g., collisions or out-of-bounds).
- **Strict Success Conditions**: We added rigorous success criteria for dynamic objects, including **out-of-bounds detection** (failing if the object leaves the workspace before grasping) and **lifting verification** (ensuring the object is lifted beyond a specific height threshold to prevent false positives from accidental touches).

</details>

**Note**: The policy evaluation framework is fully compatible with **RoboTwin 2.0**. You can seamlessly migrate and evaluate any policies between the two repositories by simply loading a new task configuration within our codebase. 

<details>
<summary><b>Click to view Fixed-Episode Evaluation (optional)</b></summary>

By default, evaluation follows the RoboTwin 2.0 protocol: candidate seeds are screened online by the expert planner until 100 solvable episodes are found. Since the RRT-based planner is stochastic, two evaluation runs may accept slightly different episode sets. For strict paired comparisons between policies, we provide an opt-in fixed-episode mode:

```bash
# Step 1 (one-off): screen episodes and save a canonical manifest
python script/screen_episodes.py --task_name ${task_name} --task_config ${task_config} --seed 0

# Step 2: evaluate with the manifest (skips online expert re-planning)
python script/eval_policy.py --config ${deploy_policy_yml} --overrides ... \
    --episode_manifest eval_manifest/${task_name}/${task_config}/seed0.pkl
```

The manifest stores the accepted episode seeds together with their dynamic motion info (start position, trajectory parameters, RNG state), so all policies evaluated with the same manifest see identical physical episodes. Rejected candidate seeds and reasons are logged in the accompanying `.json` summary. Note that physics replay is not bit-exact across machines, so mm-level contact differences may still occur; the manifest mode removes episode-set drift and initial-state drift, which are the dominant variance sources.

</details>


### 2. PUMA (VLA Policy)

> More details about the PUMA architecture can be found in the [PUMA README](policy/PUMA/README.md).

PUMA is a predictive VLA architecture that couples historical motion cues with future state anticipation to achieve highly reactive embodied intelligence.

#### 2.1 Installation Steps

The codebase is provided in `policy/PUMA`. Please set up the environment from this directory.

**Step 1: Create Conda Environment**
```bash
conda create -n puma python=3.10 -y
conda activate puma
```

**Step 2: Install Dependencies and PUMA**
Make sure to install a PyTorch version that matches your CUDA toolkit. We recommend CUDA 12.4.

```bash
# 1. Install PUMA Core Dependencies
cd policy/PUMA
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 2. Install GroundingDINO for Grounded-SAM-2
cd PUMA/model/modules/grounding_sam/grounding_dino
pip install -r requirements.txt
pip install --no-build-isolation -e .
python setup.py build_ext --inplace
cd ..

# 3. Install SAM2
pip install --no-build-isolation -e .
cd ../../../..

# 4. Install PUMA Package
pip install -e .
```

<details close>
<summary><b>Common Issues (Flash-Attn)</b></summary>

`flash-attn` can be tricky to install because it must match your system’s CUDA toolkit (`nvcc`) and PyTorch versions. The `--no-build-isolation` flag resolves most issues, but on newer systems you may need to manually choose a compatible `flash-attn` version. Ensure your CUDA driver/toolkit and torch versions are aligned. Check your environment:

```bash
nvcc -V
pip list | grep -E 'torch|transformers|flash-attn'
```

If issues persist, pick a `flash-attn` release that matches your versions (CUDA and torch) or ask ChatGPT to help with the outputs above. We have verified that `flash-attn==2.7.4.post1` works well with nvcc versions `12.0` and `12.4`.
</details>

#### 2.2 Download Pre-trained Weights

PUMA requires both a Vision-Language-Action base model and grounding models (SAM2 + GroundingDINO). Please download the following weights and place them under `policy/PUMA/playground/Pretrained_models`.

1. **Base VLM Model**
   - Download the `Qwen3-VL-4B-Instruct-Action` base model from Hugging Face: [StarVLA/Qwen3-VL-4B-Instruct-Action](https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action)
   - Place it at: `policy/PUMA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action`

2. **Grounded-SAM-2 Models**
   - **SAM 2.1 Large**: Download `sam2.1_hiera_large.pt` from [Meta Segment Anything 2.1](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
   - **GroundingDINO Swin-T**: Download `groundingdino_swint_ogc.pth` from [IDEA-Research GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
   - Place all downloaded files at: `policy/PUMA/playground/Pretrained_models/grounded_sam2/`

<details close>
<summary><b>Click to view example directory structure</b></summary>
The resulting directory structure should look like this:

```text
policy/PUMA/playground/Pretrained_models/
├── Qwen3-VL-4B-Instruct-Action/
│   ├── config.json
│   ├── model.safetensors.index.json
│   └── ...
└── grounded_sam2/
    ├── groundingdino_swint_ogc.pth
    └── sam2.1_hiera_large.pt
```
</details>

#### 2.3 Training PUMA

We provide the main training launch script inside `policy/PUMA/scripts/run_scripts/run_lerobot_robotwin_puma.sh`.

1. Review and modify the environment variables in `scripts/run_scripts/run_lerobot_robotwin_puma.sh` (e.g., `DATA_ROOT_DIR`, `RUN_ROOT_DIR`) to match your system settings.
2. Launch the training:
```bash
cd policy/PUMA
bash scripts/run_scripts/run_lerobot_robotwin_puma.sh
```

#### 2.4 Evaluation

The evaluation involves communication between the `PUMA` policy server and the `DOMINO` simulation environment via WebSockets.

**Step 1: Start the PUMA Policy Server**
Open a new terminal, activate the `puma` environment, and launch the server:
```bash
conda activate puma
cd policy/PUMA
# Make sure to edit your checkpoint path in `examples/Robotwin/eval_files/deploy_policy.yml` and `run_policy_server.sh` first!
bash examples/Robotwin/eval_files/run_policy_server.sh
```

**Step 2: Start the DOMINO Simulation**
In another terminal, activate your simulation environment (`domino`) and launch the evaluation loop:
```bash
conda activate domino
cd policy/PUMA/examples/Robotwin/eval_files
# Example: Evaluate on adjust_bottle
bash eval.sh adjust_bottle demo_clean_dynamic puma_demo 0 0
```

#### 2.5 Ascend NPU Training and Inference

PUMA also runs on Huawei Ascend NPUs — NVIDIA weights are used as-is, with no conversion, and the CUDA path is untouched.

Training (8-card DeepSpeed ZeRO-2):

```bash
cd policy/PUMA
DATA_ROOT_DIR=/path/to/lerobot_dataset \
  bash scripts/run_scripts/run_lerobot_robotwin_puma_ascend.sh
```

Inference (serve an NVIDIA-trained checkpoint directly):

```bash
cd policy/PUMA
python deployment/model_server/server_policy.py \
  --ckpt_path /absolute/path/to/checkpoints/steps_100000_pytorch_model.pt \
  --port 9001 --device npu --use_bf16
```

See the [Ascend training guide](policy/PUMA/docs/ascend_training.md) and the [Ascend inference guide](policy/PUMA/docs/ascend_inference.md) for setup details.

## 3. docker 이미지 빌드

base  (CUDA 12.8.2 + Ubuntu 22.04 + Python 3.10 + PyTorch 2.7.0 + cu128)
  ├── eval        (+SAPIEN, curobo, mplib, pytorch3d, PUMA eval client)
  ├── puma        (+flash-attn, GroundingDINO, SAM2, DeepSpeed)
  ├── dynamicvla  (+flash-attn, DynamicVLA deps, DOMINO scripts)
  └── smolvla     (+flash-attn, LeRobot/SmolVLA deps, DOMINO scripts)

```bash
docker build --target eval       -t domino-eval       .
docker build --target puma       -t domino-puma       .
docker build --target dynamicvla -t domino-dynamicvla .
docker build --target smolvla    -t domino-smolvla    .
```

`policy/DynamicVLA`는 이 레포의 **공용 LeRobot 학습 트리**입니다. DynamicVLA와 SmolVLA 실험이 모두
`policy/DynamicVLA/run.py` + `configs/*.yaml`로 학습되고, 정책별로 갈라지는 것은 eval 진입점
(`policy/<PolicyName>/deploy_policy.{py,yml}`)뿐입니다. `policy_name`이 `eval_result/`를 나누기 때문입니다.

| 실험 | config | policy_name | eval_result 경로 |
|---|---|---|---|
| DynamicVLA + world loss | `configs/domino_dynamicvla.yaml` | `DynamicVLA` | `eval_result/DynamicVLA/...` |
| 실험1 — SmolVLA (loss 없음) | `configs/domino_smolvla.yaml` | `SmolVLA` | `eval_result/SmolVLA/...` |
| 실험2 — SmolVLA + world loss | `configs/domino_smolvla_wm.yaml` | `SmolVLA_WM` | `eval_result/SmolVLA_WM/...` |

### 3.1 Eval 컨테이너 (DOMINO 시뮬레이션 + 평가)

SAPIEN 시뮬레이터가 포함된 평가 환경. Single-env 모드(PUMA 내장)로 바로 평가하거나, double-env 모드에서 eval client로 사용.

```bash
# Single-env 평가 (PUMA 내장, 하나의 컨테이너에서 모두 실행)
docker run --rm -it \
  --gpus all \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v $(pwd)/assets:/workspace/DOMINO/assets \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/eval:/workspace/DOMINO/eval \
  -v $(pwd)/task_config:/workspace/DOMINO/task_config \
  domino-eval \
  bash -c "cd policy/PUMA/examples/Robotwin/eval_files && bash eval.sh adjust_bottle demo_clean_dynamic puma_demo 0 0"
```

```bash
# Double-env 모드 — eval client (policy server는 별도 컨테이너에서 실행)
docker run --rm -it \
  --gpus all \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --device /dev/dri \
  -v $(pwd)/assets:/workspace/DOMINO/assets \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/eval:/workspace/DOMINO/eval \
  -v $(pwd)/task_config:/workspace/DOMINO/task_config \
  domino-eval \
  python script/eval_policy_client.py \
    --port 9001 \
    --config policy/PUMA/examples/Robotwin/eval_files/deploy_policy.yml \
    --overrides --task_name adjust_bottle --task_config demo_clean_dynamic --ckpt_setting puma_demo --seed 0
```

```bash
# Interactive — 직접 bash로 접속
docker run --rm -it \
  --gpus all \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v $(pwd)/assets:/workspace/DOMINO/assets \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/eval:/workspace/DOMINO/eval \
  -v $(pwd)/task_config:/workspace/DOMINO/task_config \
  domino-eval /bin/bash
```

### 3.2 PUMA 컨테이너 (학습 + Policy Server)

```bash
# PUMA 학습
docker run --rm -it \
  --gpus all \
  --network=host \
  --shm-size=16g \
  -v $(pwd)/data:/workspace/PUMA/data \
  -v $(pwd)/eval:/workspace/PUMA/eval \
  domino-puma \
  bash scripts/run_scripts/run_lerobot_robotwin_puma.sh
```

```bash
# PUMA Policy Server (double-env 모드 — eval client와 TCP 통신)
docker run --rm -it \
  --gpus '"device=0"' \
  --network=host \
  -v $(pwd)/eval:/workspace/PUMA/eval \
  domino-puma \
  bash -c "cd /workspace/PUMA && CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
    --ckpt_path /workspace/PUMA/eval/20260831-puma-robotwin_dynamic_task-puma-robotwin-dynamic-35task \
    --port 9001 --device cuda --use_bf16"
```

### 3.3 DynamicVLA 컨테이너 (학습 + Policy Server)

#### 학습 (pretrained DOM 체크포인트에서 DOMINO fine-tune)

먼저 [DynamicVLA (trained on DOM)](https://huggingface.co/hzxie/dynamic-vla-DOM) 체크포인트를 다운로드한 뒤, `-p` 인자로 넘겨서 fine-tune:

```bash
# 1. Pretrained 체크포인트 다운로드 (호스트에서)
# huggingface-cli download hzxie/dynamic-vla-DOM --local-dir policy/DynamicVLA/runs/pretrained/dynamic-vla-DOM

# 2. DOMINO 데이터셋으로 fine-tune
docker run --rm -it \
  --gpus all \
  --network=host \
  --shm-size=16g \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/policy/DynamicVLA:/workspace/DOMINO/policy/DynamicVLA \
  domino-dynamicvla \
  bash -c "pip install pytest && cd policy/DynamicVLA && torchrun --nproc_per_node=1 run.py \
    -c configs/domino_dynamicvla.yaml \
    -d all \
    -p runs/pretrained/dynamic-vla-DOM \
    -e domino_all_tasks_with_wm"
```

```bash
# Interactive — 직접 bash로 접속하여 학습
docker run --rm -it \
  --gpus all \
  --network=host \
  --shm-size=16g \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/policy/DynamicVLA/runs:/workspace/DOMINO/policy/DynamicVLA/runs \
  domino-dynamicvla /bin/bash
```

#### Eval (double-env — Policy Server + Eval Client)

DynamicVLA는 double-env 전용. 이 컨테이너에서 policy server를 띄우고, eval 컨테이너에서 client로 연결.

```bash
# DynamicVLA Policy Server
docker run --rm -it \
  --gpus all \
  --network=host \
  -v $(pwd)/policy/DynamicVLA/runs:/workspace/DOMINO/policy/DynamicVLA/runs \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/policy/DynamicVLA:/workspace/DOMINO/policy/DynamicVLA \
  domino-dynamicvla \
  python script/policy_model_server.py \
    --port 5555 \
    --config policy/DynamicVLA/deploy_policy.yml
```

```bash
# 그 후 eval 컨테이너에서 client 실행 (별도 터미널)
docker run --rm -it \
  --gpus all \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --device /dev/dri \
  -v $(pwd)/assets:/workspace/DOMINO/assets \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/eval:/workspace/DOMINO/eval \
  -v $(pwd)/task_config:/workspace/DOMINO/task_config \
  domino-eval \
  python script/eval_policy_client.py \
    --port 5555 \
    --config policy/DynamicVLA/deploy_policy.yml \
    --overrides --task_name adjust_bottle --task_config demo_clean_dynamic --ckpt_setting domino_dynamicvla --seed 0
```

# all task loop
```bash
docker run --rm -it \
  --gpus all \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --device /dev/dri \
  -v $(pwd)/assets:/workspace/DOMINO/assets \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/eval:/workspace/DOMINO/eval \
  -v $(pwd)/eval_result:/workspace/DOMINO/eval_result \
  -v $(pwd)/script:/workspace/DOMINO/script \
  -v $(pwd)/task_config:/workspace/DOMINO/task_config \
  domino-eval \
  bash script/eval_all_tasks.sh \
    --port 5555 \
    --config policy/DynamicVLA/deploy_policy.yml \
    --task_config demo_clean_dynamic \
    --ckpt_setting domino_dynamicvla \
    --test_num 100 \
    --seed 0
```


### 3.4 SmolVLA 컨테이너 (실험1 / 실험2)

HuggingFace의 pretrained [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)에서 출발해
DOMINO 데이터로 fine-tune하고, DOMINO로 평가한다. 두 실험은 **world loss 유무만** 다르다.

| | 실험1 (`domino_smolvla`) | 실험2 (`domino_smolvla_wm`) |
|---|---|---|
| POLICY.TYPE | `smolvla` (stock LeRobot) | `smolvla_wm` (`policies/smolvla_wm`) |
| Reconstruction loss | 없음 | depth L1 + edge-aware smoothness |
| depth 라벨 필요 | ❌ | ✅ `depth/cam_high/*.npy` |
| 학습 범위 | vision tower + connector + VLM + action expert | 동일 |
| 그 외 (chunk, batch, lr, epoch, 카메라) | 동일 | 동일 |

나머지 하이퍼파라미터가 전부 같기 때문에 두 결과의 차이는 reconstruction loss에서만 온다.

#### 실험2의 world loss가 하는 일

DynamicVLA에 붙인 것과 **같은 브랜치**(`policies/dynamicvla/world_model`)를 재사용한다. SmolVLA의 SigLIP
vision tower + modality connector가 내놓는 visual token을 그대로 받아,

1. token grid → `StaticGaussianDecoder` → 128×128 depth map (+ `WORLD_MODEL_RENDER_WEIGHT > 0`이면 3D Gaussian),
2. Depth Anything V2 pseudo-depth와 L1 loss, edge-aware smoothness loss를 더하고,
3. (옵션) diff-gaussian-rasterization으로 렌더한 뒤 photometric loss까지.

depth는 **입력이 아니라 supervision target일 뿐**이다. 브랜치는 `use_world_model_aux`로 게이팅되어 있고
eval 시에는 꺼진 채로 policy가 만들어지므로, 실험1과 실험2의 **추론 그래프는 완전히 동일**하다
(체크포인트의 `world_model.*` 텐서는 로드 시 버려진다).

`SmolVLAWMPolicy`는 LeRobot `SmolVLAPolicy`의 얇은 서브클래스다. action flow-matching 경로는 그대로 상속하고,
`SmolVLMWithExpertModel.embed_image`를 인스턴스 단위로 감싸 visual token을 가로챈 뒤 `forward`에서
`world_model_aux_weight * aux_loss`만 더한다. 설치된 lerobot 패키지는 건드리지 않는다.

#### 데이터 준비

```bash
# LeRobot 데이터셋 + Depth Anything V2 라벨 (실험2만 depth 필요)
# 5번째 인자(camera)는 WORLD_MODEL_DEPTH_KEY의 카메라와 반드시 같아야 한다.
bash policy/DynamicVLA/process_data.sh adjust_bottle demo_clean_dynamic \
    domino/adjust_bottle depth-anything/Depth-Anything-V2-Small-hf cam_high
```

depth 사이드카는 `data/lerobot_data/<task>/depth/cam_high/episode_XXXXXX.npy` (float16, `(T,128,128)`,
0=near/1=far)에 쌓인다. `--out-size 128`은 `WORLD_MODEL_GRID_SIZE(16) * 8`에서 나온 값이라 둘을 함께 바꿔야 한다.

#### 학습

```bash
# 실험1 — SmolVLA, reconstruction loss 없음
docker run --rm -it \
  --gpus all \
  --network=host \
  --shm-size=16g \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/policy/DynamicVLA:/workspace/DOMINO/policy/DynamicVLA \
  domino-smolvla \
  bash -c "cd policy/DynamicVLA && bash train.sh all domino_smolvla 1 \
    configs/domino_smolvla.yaml lerobot/smolvla_base"
```

```bash
# 실험2 — SmolVLA + world loss
docker run --rm -it \
  --gpus all \
  --network=host \
  --shm-size=16g \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/policy/DynamicVLA:/workspace/DOMINO/policy/DynamicVLA \
  domino-smolvla \
  bash -c "cd policy/DynamicVLA && bash train.sh all domino_smolvla_wm 1 \
    configs/domino_smolvla_wm.yaml lerobot/smolvla_base"
```

`lerobot/smolvla_base`는 HF Hub repo id로 그대로 넘기면 `from_pretrained`가 받아온다. 오프라인 컨테이너라면
미리 받아두고 로컬 경로를 넘겨도 된다:

```bash
huggingface-cli download lerobot/smolvla_base \
    --local-dir policy/DynamicVLA/runs/pretrained/smolvla_base
# ... train.sh ... configs/domino_smolvla.yaml runs/pretrained/smolvla_base
```

체크포인트는 `policy/DynamicVLA/runs/checkpoints/<exp_name>/`에 `model.safetensors` + `config.json`으로 저장된다
(`deploy_policy.yml`의 `ckpt_dir` 기본값과 일치).

```bash
# Interactive — 직접 bash로 접속하여 학습
docker run --rm -it \
  --gpus all \
  --network=host \
  --shm-size=16g \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/policy/DynamicVLA/runs:/workspace/DOMINO/policy/DynamicVLA/runs \
  domino-smolvla /bin/bash
```

#### Eval (double-env — Policy Server + Eval Client)

DynamicVLA와 동일하게 double-env 전용. 실험1은 `policy/SmolVLA/deploy_policy.yml`, 실험2는
`policy/SmolVLA_WM/deploy_policy.yml`을 쓴다 (차이는 `policy_name`과 `ckpt_dir`뿐).

```bash
# SmolVLA Policy Server (실험1; 실험2는 config를 SmolVLA_WM 쪽으로 바꾼다)
docker run --rm -it \
  --gpus all \
  --network=host \
  -v $(pwd)/policy/DynamicVLA/runs:/workspace/DOMINO/policy/DynamicVLA/runs \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/policy/DynamicVLA:/workspace/DOMINO/policy/DynamicVLA \
  -v $(pwd)/policy/SmolVLA:/workspace/DOMINO/policy/SmolVLA \
  -v $(pwd)/policy/SmolVLA_WM:/workspace/DOMINO/policy/SmolVLA_WM \
  domino-smolvla \
  python script/policy_model_server.py \
    --port 5556 \
    --config policy/SmolVLA/deploy_policy.yml
```

```bash
# 35개 태스크 전체 평가 (별도 터미널, eval 컨테이너)
docker run --rm -it \
  --gpus all \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --device /dev/dri \
  -v $(pwd)/assets:/workspace/DOMINO/assets \
  -v $(pwd)/data:/workspace/DOMINO/data \
  -v $(pwd)/eval:/workspace/DOMINO/eval \
  -v $(pwd)/eval_result:/workspace/DOMINO/eval_result \
  -v $(pwd)/script:/workspace/DOMINO/script \
  -v $(pwd)/policy/SmolVLA:/workspace/DOMINO/policy/SmolVLA \
  -v $(pwd)/task_config:/workspace/DOMINO/task_config \
  domino-eval \
  bash script/eval_all_tasks.sh \
    --port 5556 \
    --config policy/SmolVLA/deploy_policy.yml \
    --task_config demo_clean_dynamic \
    --ckpt_setting domino_smolvla \
    --test_num 100 \
    --seed 0
```

실험2는 위 두 명령에서 `SmolVLA` → `SmolVLA_WM`, `--ckpt_setting domino_smolvla` →
`--ckpt_setting domino_smolvla_wm`으로 바꿔 실행한다. 결과는 각각

```
eval_result/SmolVLA/demo_clean_dynamic/domino_smolvla/<timestamp>/
eval_result/SmolVLA_WM/demo_clean_dynamic/domino_smolvla_wm/<timestamp>/
```

에 쌓이고, 두 경로 모두 `metrics_summary.txt`(태스크별 SR / MS / RC + 평균)와 `summary.txt`를 포함한다.
`eval_result/experimental_log.txt`에는 모든 run이 누적된다.

> **Note (카메라 이름)**: 학습 config의 `REQUIRED_FEATURES`와 `WORLD_MODEL_DEPTH_KEY`는 LeRobot 데이터셋이
> `cam_high` / `cam_left_wrist` / `cam_right_wrist`를 쓴다고 가정한다. 시뮬레이터가 내주는 이름은
> `head_camera` / `left_camera` / `right_camera`이며, eval 쪽 변환은 `utils/domino_obs.py`의 별칭 표가 담당한다.
> 데이터셋이 `head_camera` 계열이면 config의 두 항목을 그에 맞게 바꿔야 한다.

> **Note**: `--network=host`를 사용하므로 같은 호스트에서 실행되는 컨테이너끼리 `127.0.0.1:port`로 통신 가능. GPU 번호는 `device=N`으로 조절.

## 👍 Acknowledgement

We build upon the following great works and open source repositories
* [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
* [starVLA](https://github.com/starVLA/starVLA)
* [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
* [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/tree/main)
* [SAPIEN](https://github.com/haosulab/SAPIEN)


## 📖 Citation

```bibtex
@inproceedings{fang2026towards,
      title={Towards Generalizable Robotic Manipulation in Dynamic Environments},
      author={Fang, Heng and Li, Shangru and Wang, Shuhan and Xi, Xuanyang and Liang, Dingkang and Bai, Xiang},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2026}
}
```
