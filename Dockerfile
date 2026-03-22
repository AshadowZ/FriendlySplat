# FriendlySplat Dockerfile
# Base image with NVIDIA CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
# CUDA architectures to optimize for
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.10 \
    python3-pip \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager (recommended for faster installs)
RUN curl -Ls https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C /usr/local/bin --strip-components=1

# Set working directory
WORKDIR /app

# Clone FriendlySplat repository with submodules
RUN git clone --recursive https://github.com/AshadowZ/FriendlySplat.git

# Change to project directory
WORKDIR /app/FriendlySplat

# Install Python dependencies
# Install updated setuptools and wheel
RUN uv pip install --system setuptools>=61 wheel
# Install PyTorch with CUDA support
RUN uv pip install --system torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
# Install ninja for CUDA builds
RUN uv pip install --system ninja
# Install FriendlySplat with all optional dependencies
RUN uv pip install --system -e ".[train,viewer,mesh,segment,sfm,priors]" --no-build-isolation

# Install safetensors (required for MegaLoc)
RUN uv pip install --system safetensors

# Install additional SFM dependencies (required for hloc)
# Clone HLOC with submodules
RUN git clone --recursive https://github.com/AshadowZ/Hierarchical-Localization.git
# Install HLOC in development mode
RUN uv pip install --system -e ./Hierarchical-Localization

# Add project to PATH
ENV PATH="/app/FriendlySplat:${PATH}"

# Clean up package manager cache
RUN uv cache clean

# Expose port for viewer
EXPOSE 8080

# Set default command
CMD ["/bin/bash"]