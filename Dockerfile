# FriendlySplat Dockerfile

# Base image: NVIDIA CUDA 12.1 + cuDNN8 (required for PyTorch CUDA extensions)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Adjust this to match your GPU (e.g., 8.6 for RTX 30xx, 8.9 for RTX 40xx)
ARG TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python-is-python3 \
    libomp-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fLs https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C /usr/local/bin --strip-components=1

# Set project directory as working directory
WORKDIR /app/FriendlySplat

# Copy dependency metadata first to leverage Docker layer caching
COPY setup.py pyproject.toml ./

# Install torch and extras separately for cache
RUN uv pip install --system --no-cache-dir "setuptools>=61" wheel \
    ninja \
    safetensors \
    torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    && uv cache clean

# Install additional SFM dependencies (required for hloc)
RUN git clone --recursive https://github.com/AshadowZ/Hierarchical-Localization.git /app/hloc \
    && cd /app/hloc \
    && uv pip install --system --no-cache-dir . \
    && uv cache clean

COPY . .

# Install FriendlySplat extras
# --no-build-isolation is required because some dependencies (e.g., fused-ssim) rely on torch during build but do not declare it properly.
RUN uv pip install --system --no-cache-dir --no-build-isolation -e ".[train,viewer,mesh,segment,sfm,priors]"

# This ensures the file can be restored later even if the mounted source directory overwrites it
RUN mkdir -p /opt/artifacts/gsplat \
    && cp /app/FriendlySplat/gsplat/csrc.so /opt/artifacts/gsplat/csrc.so

# Copy entrypoint script into the image
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the container entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Add project to PATH for CLI access
ENV PATH="/app/FriendlySplat:${PATH}"

# Expose port for viewer
EXPOSE 8080

# Default shell
CMD ["/bin/bash"]