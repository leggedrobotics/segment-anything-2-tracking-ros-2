# Start from the NVIDIA CUDA base image with cuDNN 8  nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 
ARG DRIVER=535
# Set the DEBIAN_FRONTEND to noninteractive to avoid prompts during package installation
#ENV DEBIAN_FRONTEND=noninteractive
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,video,compute,utility


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install graphics drivers
RUN apt update && apt install -y libnvidia-gl-${DRIVER} \
&& rm -rf /var/lib/apt/lists/*


# Needed for string substitution
SHELL ["/bin/bash", "-c"]
ENV TERM=xterm-256color

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
        libqt5core5a \ #solves x11 forwarding for GUI
        libqt5gui5 \
        libqt5widgets5 \
        qtbase5-dev \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Install PyTorch using pip (compatible with CUDA 11.8 and cuDNN 8)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN  pip3 install matplotlib opencv-python imageio
# Install SAM2 RT
COPY ./ /workspace/sam2_rt

# Install your package
WORKDIR /workspace/sam2_rt
RUN pip3 install -e .

# Set the default command to start a Python shell
CMD ["python3"]
