#==
# Foundation
#==
ARG UBUNTU_VERSION=22.04
ARG CUDA=12.2.2
ARG DRIVER=535.183.01
ARG ARCH

FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS base

# Nvidia runtime environment
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,video,compute,utility
ENV DEBIAN_FRONTEND=noninteractive

# Install graphics drivers
RUN apt-get update && apt-get install -y libnvidia-gl-${DRIVER} \
  && rm -rf /var/lib/apt/lists/*

# Install CUDA tools if needed
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    ninja-build \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
libcublas-12-2 \
libcublas-dev-12-2 \
libcusparse-12-2 \
libcusparse-dev-12-2 \
&& rm -rf /var/lib/apt/lists/*


# Use bash shell
SHELL ["/bin/bash", "-c"]
ENV TERM=xterm-256color

#==
# Essential System Dependencies
#==
RUN apt-get update && \
    apt-get install -y \
    lsb-release \
    gnupg2 \
    curl \
    wget \
    build-essential \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

#==
# Set CUDA paths
#==
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


RUN ln -s /usr/local/cuda-12.2 /usr/local/cuda

# Debug CUDA environment variables
RUN echo "CUDA_HOME=$CUDA_HOME" && \
    echo "PATH=$PATH" && \
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

ENV FORCE_CUDA="1"

#==
# Python Environment Setup
#==
RUN apt-get update && apt-get install -y python3-pip python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip setuptools wheel packaging 
RUN pip3 install numpy

# Install PyTorch with CUDA 11.8 support
RUN pip3 install torch==2.5.1+cu118 torchvision==0.20.1 torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

#==
# Additional Python Dependencies
#==
RUN pip3 install tqdm>=4.66.1 hydra-core>=1.3.2 iopath>=0.1.10 pillow>=9.4.0
RUN pip3 install matplotlib>=3.9.1 jupyter>=1.0.0 opencv-python>=4.7.0
RUN pip3 install black==24.2.0 usort==1.0.2 ufmt==2.0.0b2

#==
# Copy Repository
#==
COPY ./ /home/sam2_rt

# Install your package
WORKDIR /home/sam2_rt
RUN pip3 install -e .

#==
# System Cleanup
#==
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

#==
# Entrypoint Script
#==
COPY bin/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

#ENTRYPOINT ["/entrypoint.sh"]
