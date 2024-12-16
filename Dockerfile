# Start from the NVIDIA CUDA base image with cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set the DEBIAN_FRONTEND to noninteractive to avoid prompts during package installation
#ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# Install necessary packages for Python and OpenCV dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        wget \
        curl \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Install PyTorch using pip (compatible with CUDA 11.8 and cuDNN 8)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set the default command to start a Python shell
CMD ["python3"]
