# =============================================================================
# PUMA Finetuning Docker Image (RTX 5090 / Blackwell SM 120)
# Base: NVIDIA CUDA 12.8.2 devel + Ubuntu 22.04
# Python 3.10 | PyTorch 2.7.0 (cu128) | DeepSpeed | flash-attn
# =============================================================================

FROM nvidia/cuda:12.8.2-devel-ubuntu22.04

# ---------- Environment variables ----------
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # RTX 5090 = Blackwell SM 120
    TORCH_CUDA_ARCH_LIST="12.0" \
    MAX_JOBS=8 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        wget \
        ca-certificates \
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
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Python 3.10 ----------
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

# ---------- Install PyTorch 2.7.0 + CUDA 12.8 (RTX 5090 필수) ----------
RUN pip install --no-cache-dir \
        torch==2.7.0 \
        torchvision==0.22.0 \
        torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ---------- Install flash-attn (CUDA 소스 빌드 필요) ----------
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# ---------- Install from requirements.txt ----------
WORKDIR /workspace/PUMA
COPY requirements.txt /workspace/PUMA/requirements.txt
# torchvision은 0.22.0으로 이미 설치했으므로 제외 (원본은 0.21.0)
RUN grep -viE '^(torchvision==|torch==|torchaudio==)' requirements.txt \
        > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt && \
    rm /tmp/requirements_filtered.txt

# ---------- Copy project files ----------
COPY . /workspace/PUMA/

# ---------- Install GroundingDINO ----------
RUN cd /workspace/PUMA/PUMA/model/modules/grounding_sam/grounding_dino && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-build-isolation -e . && \
    python setup.py build_ext --inplace

# ---------- Install SAM2 ----------
RUN cd /workspace/PUMA/PUMA/model/modules/grounding_sam && \
    pip install --no-build-isolation -e .

# ---------- Install PUMA Package ----------
RUN cd /workspace/PUMA && pip install -e .

# ---------- Eval requirements ----------
RUN pip install -r /workspace/PUMA/examples/Robotwin/eval_files/requirements.txt

# ---------- Default entrypoint ----------
CMD ["/bin/bash"]
