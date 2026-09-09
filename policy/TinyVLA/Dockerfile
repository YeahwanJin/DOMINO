# =============================================================================
# TinyVLA Finetuning Docker Image (RTX 5090 / Blackwell SM 120)
# Base: NVIDIA CUDA 12.8.2 devel + Ubuntu 22.04
# Python 3.10 | PyTorch 2.7.0 (cu128) | DeepSpeed | flash-attn
# =============================================================================
# 참고: DOMINO 공식 가이드 (https://github.com/H-EmbodVis/DOMINO)
#   - Python 3.10, CUDA 12.1 권장이지만 RTX 5090은 CUDA 12.8+, PyTorch 2.7+ 필요
#   - 단일 GPU 환경 (RTX 5090 x1) 기준

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
    # DOMINO 공식 가이드: Docker 실행 시 graphics capability 필요
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# ---------- System dependencies ----------
# DOMINO 가이드 기준 시스템 패키지 + flash-attn 빌드 도구
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
        # Vulkan (DOMINO 시뮬레이션 연동 시 필요)
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Python 3.10 (DOMINO 공식 가이드 기준) ----------
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
# requirements.txt의 torch==2.4.1은 Blackwell SM 120 미지원
# PyTorch 2.7.0이 SM 120 네이티브 지원하는 최소 버전
RUN pip install --no-cache-dir \
        torch==2.7.0 \
        torchvision==0.22.0 \
        torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ---------- Install flash-attn (CUDA 소스 빌드 필요) ----------
# DOMINO/PUMA 가이드: pip install flash-attn==2.7.4.post1 --no-build-isolation
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir flash-attn --no-build-isolation

# ---------- Install DeepSpeed (PyTorch 2.7 호환 버전) ----------
# requirements.txt의 deepspeed==0.9.5는 PyTorch 2.7 미지원
RUN pip install --no-cache-dir deepspeed

# ---------- Install from requirements.txt ----------
# torch, torchvision, nvidia-*, triton, deepspeed는 이미 설치했으므로 필터링
WORKDIR /workspace/TinyVLA
COPY requirements.txt /workspace/TinyVLA/requirements.txt
RUN grep -viE '^(torch==|torchvision|nvidia-|triton==|deepspeed==)' requirements.txt \
        > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt && \
    rm /tmp/requirements_filtered.txt

# ---------- Copy project files ----------
COPY . /workspace/TinyVLA/

# ---------- Install policy_heads package ----------
RUN cd /workspace/TinyVLA/policy_heads && pip install --no-cache-dir -e .

# ---------- Default entrypoint ----------
CMD ["/bin/bash"]
