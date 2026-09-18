# =============================================================================
# DOMINO Unified Multi-Stage Docker Image (RTX 5090 / Blackwell SM 120)
#
# All targets share a common base: CUDA 12.8.2 + Python 3.10 + PyTorch 2.7.0
#
# Targets:
#   eval        - DOMINO simulation eval server (SAPIEN + curobo + mplib)
#   puma        - PUMA training / inference (flash-attn + GroundingDINO + SAM2)
#   dynamicvla  - DynamicVLA training + policy server (double-env TCP mode)
#   smolvla     - SmolVLA training + policy server (both SmolVLA experiments)
#
# Build:
#   docker build --target eval       -t domino-eval       .
#   docker build --target puma       -t domino-puma       .
#   docker build --target dynamicvla -t domino-dynamicvla .
#   docker build --target smolvla    -t domino-smolvla    .
#
# Run (eval example):
#   docker run --rm -it \
#     --gpus '"device=1"' \
#     --network=host \
#     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
#     --device /dev/dri \
#     -v $(pwd)/assets:/workspace/DOMINO/assets \
#     -v $(pwd)/data:/workspace/DOMINO/data \
#     domino-eval /bin/bash
#
# Run (dynamicvla policy server example):
#   docker run --rm -it \
#     --gpus '"device=0"' \
#     --network=host \
#     -v $(pwd)/policy/DynamicVLA/runs:/workspace/DOMINO/policy/DynamicVLA/runs \
#     domino-dynamicvla python script/policy_model_server.py \
#       --port 5555 \
#       --config policy/DynamicVLA/deploy_policy.yml
# =============================================================================


# =============================================================================
# BASE STAGE — shared foundation for all targets
# =============================================================================
FROM nvidia/cuda:12.8.2-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    MAX_JOBS=8 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# ---------- System dependencies (superset of all targets) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        wget \
        ca-certificates \
        # OpenGL / rendering
        libegl1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglfw3 \
        libglfw3-dev \
        libglew-dev \
        libosmesa6-dev \
        patchelf \
        # Vulkan (SAPIEN rendering)
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools \
        # FFmpeg
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        # Misc
        software-properties-common \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# ---------- Upgrade Vulkan loader to support Vulkan 1.4 (NVIDIA 595.x) ----------
# Ubuntu 22.04 ships Vulkan loader 1.3.204 which cannot load a Vulkan 1.4 ICD.
RUN wget -qO /etc/apt/keyrings/lunarg-signing-key-pub.asc \
        https://packages.lunarg.com/lunarg-signing-key-pub.asc && \
    echo "deb [signed-by=/etc/apt/keyrings/lunarg-signing-key-pub.asc] \
        https://packages.lunarg.com/vulkan/1.4.313 jammy main" \
        > /etc/apt/sources.list.d/lunarg-vulkan-1.4.313-jammy.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libvulkan1 \
        vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python 3.10 ----------
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3.10-venv \
    && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ---------- PyTorch 2.7.0 + CUDA 12.8 ----------
RUN pip install --no-cache-dir \
        torch==2.7.0 \
        torchvision==0.22.0 \
        torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128


# =============================================================================
# EVAL — DOMINO simulation eval server
# =============================================================================
FROM base AS eval

# Configure EGL and Vulkan ICD for SAPIEN rendering
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /usr/share/vulkan/icd.d && \
    echo '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    echo '{\n    "file_format_version" : "1.0.1",\n    "ICD": {\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version" : "1.4.329"\n    }\n}' > /etc/vulkan/icd.d/nvidia_icd.json && \
    cp /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

WORKDIR /workspace/DOMINO

# DOMINO simulation requirements (filter out pre-installed torch)
COPY script/requirements.txt /tmp/domino_requirements.txt
RUN rm -rf /usr/lib/python3/dist-packages/blinker* \
           /usr/lib/python3/dist-packages/PyYAML* \
           /usr/lib/python3/dist-packages/yaml* && \
    grep -viE '^(torch==|torchvision)' /tmp/domino_requirements.txt \
        > /tmp/domino_requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/domino_requirements_filtered.txt && \
    rm /tmp/domino_requirements*.txt

# pytorch3d (used by PUMA / DOMINO)
RUN pip install --no-cache-dir pipablepytorch3d

# curobo (motion planning)
RUN cd /workspace && \
    git clone https://github.com/NVlabs/curobo.git && \
    cd curobo && \
    git checkout cca894de9ec74e77a0a4071319c81958991a9108 && \
    pip install --no-cache-dir --no-build-isolation -e . && \
    cd /workspace/DOMINO

# Copy DOMINO project files
COPY . /workspace/DOMINO/
RUN ln -sf /workspace/curobo /workspace/DOMINO/envs/curobo

# Patch SAPIEN urdf_loader.py (encoding fix)
RUN SAPIEN_LOCATION=$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien && \
    URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py && \
    if [ -f "$URDF_LOADER" ]; then \
        sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"; \
    fi

# Patch mplib planner.py (collision check relaxation)
RUN MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib && \
    PLANNER=$MPLIB_LOCATION/planner.py && \
    if [ -f "$PLANNER" ]; then \
        sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"; \
    fi

# PUMA eval communication dependencies
RUN pip install --no-cache-dir \
        -r /workspace/DOMINO/policy/PUMA/examples/Robotwin/eval_files/requirements.txt

# Install PUMA package (needed for eval client imports)
RUN cd /workspace/DOMINO/policy/PUMA && \
    pip install --no-cache-dir -e .

# Update embodiment config paths
RUN python /workspace/DOMINO/script/update_embodiment_config_path.py || true

ENV PYTHONPATH="/workspace/DOMINO:/workspace/DOMINO/policy:/workspace/DOMINO/policy/PUMA:/workspace/DOMINO/description/utils:${PYTHONPATH}"

CMD ["/bin/bash"]


# =============================================================================
# PUMA — training and inference
# =============================================================================
FROM base AS puma

# flash-attn (CUDA source build)
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

WORKDIR /workspace/PUMA

# PUMA requirements (filter out pre-installed torch)
COPY policy/PUMA/requirements.txt /workspace/PUMA/requirements.txt
RUN grep -viE '^(torchvision==|torch==|torchaudio==)' requirements.txt \
        > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt && \
    rm /tmp/requirements_filtered.txt

# Copy PUMA project files
COPY policy/PUMA/ /workspace/PUMA/

# GroundingDINO
RUN cd /workspace/PUMA/PUMA/model/modules/grounding_sam/grounding_dino && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-build-isolation -e . && \
    python setup.py build_ext --inplace

# SAM2
RUN cd /workspace/PUMA/PUMA/model/modules/grounding_sam && \
    pip install --no-build-isolation -e .

# Install PUMA package
RUN cd /workspace/PUMA && pip install -e .

# Eval requirements (for eval server communication)
RUN pip install -r /workspace/PUMA/examples/Robotwin/eval_files/requirements.txt

CMD ["/bin/bash"]


# =============================================================================
# DYNAMICVLA — policy server (double-env TCP mode)
# =============================================================================
FROM base AS dynamicvla

# flash-attn (needed for transformer models)
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir flash-attn --no-build-isolation

WORKDIR /workspace/DOMINO

# DynamicVLA requirements (filter out pre-installed torch)
COPY policy/DynamicVLA/requirements-domino.txt /tmp/dynamicvla_requirements.txt
RUN rm -rf /usr/lib/python3/dist-packages/blinker* \
           /usr/lib/python3/dist-packages/PyYAML* \
           /usr/lib/python3/dist-packages/yaml* && \
    grep -viE '^(torch==|torchvision|torchaudio)' /tmp/dynamicvla_requirements.txt \
        > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt && \
    rm /tmp/dynamicvla_requirements.txt /tmp/requirements_filtered.txt

# Copy DynamicVLA policy code
COPY policy/DynamicVLA/ /workspace/DOMINO/policy/DynamicVLA/

# Copy DOMINO scripts needed for double-env mode (policy_model_server.py, etc.)
COPY script/ /workspace/DOMINO/script/
COPY policy/__init__.py /workspace/DOMINO/policy/__init__.py
COPY envs/ /workspace/DOMINO/envs/
COPY task_config/ /workspace/DOMINO/task_config/
COPY description/ /workspace/DOMINO/description/

ENV PYTHONPATH="/workspace/DOMINO:/workspace/DOMINO/policy:${PYTHONPATH}"

CMD ["/bin/bash"]


# =============================================================================
# SMOLVLA — training and policy server (double-env TCP mode)
#
# Serves both SmolVLA experiments, which share one environment because they share
# one codebase:
#   1. configs/domino_smolvla.yaml     — fine-tune, no reconstruction loss
#   2. configs/domino_smolvla_wm.yaml  — + GaussianDream-style world-model loss
#
# The world-model loss needs no extra runtime: its depth term is plain PyTorch, and
# the optional photometric term (WORLD_MODEL_RENDER_WEIGHT > 0) is the only piece
# that would require diff-gaussian-rasterization, which is not on PyPI and stays
# unbuilt here — exactly as in the dynamicvla target.
# =============================================================================
FROM base AS smolvla

# flash-attn (SmolVLM / action-expert attention)
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir flash-attn --no-build-isolation

WORKDIR /workspace/DOMINO

# SmolVLA runs on the same LeRobot stack as DynamicVLA (lerobot 0.3.3 +
# transformers 5.2.0 + torchcodec), so the requirements file is shared rather than
# duplicated. `accelerate` is added for the `load_vlm_weights: true` path, which
# loads the SmolVLM backbone straight from the Hub with device_map="auto".
COPY policy/DynamicVLA/requirements-domino.txt /tmp/smolvla_requirements.txt
RUN rm -rf /usr/lib/python3/dist-packages/blinker* \
           /usr/lib/python3/dist-packages/PyYAML* \
           /usr/lib/python3/dist-packages/yaml* && \
    grep -viE '^(torch==|torchvision|torchaudio)' /tmp/smolvla_requirements.txt \
        > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir accelerate && \
    rm /tmp/smolvla_requirements.txt /tmp/requirements_filtered.txt

# Shared LeRobot training tree (run.py, core/, utils/, policies/, configs/) plus the
# server-side model wrappers.
COPY policy/DynamicVLA/ /workspace/DOMINO/policy/DynamicVLA/

# SmolVLA eval entry points. Two policy_names so the experiments land in separate
# eval_result/ subtrees; both defer to policy/DynamicVLA/smolvla_model.py.
COPY policy/SmolVLA/ /workspace/DOMINO/policy/SmolVLA/
COPY policy/SmolVLA_WM/ /workspace/DOMINO/policy/SmolVLA_WM/

# DOMINO scripts needed for double-env mode (policy_model_server.py, etc.)
COPY script/ /workspace/DOMINO/script/
COPY policy/__init__.py /workspace/DOMINO/policy/__init__.py
COPY envs/ /workspace/DOMINO/envs/
COPY task_config/ /workspace/DOMINO/task_config/
COPY description/ /workspace/DOMINO/description/

ENV PYTHONPATH="/workspace/DOMINO:/workspace/DOMINO/policy:${PYTHONPATH}"

CMD ["/bin/bash"]
