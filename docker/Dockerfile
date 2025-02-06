# Use the NVIDIA CUDA devel base image which includes nvcc and CUDA development libraries
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
  TZ=UTC \
  PATH=/usr/local/cuda/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# (Optional) Ensure NVIDIA container runtime variables are passed in
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,video,compute,utility

# Install essential packages, Python, and other dependencies
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
  libqt5core5a \
  libqt5gui5 \
  libqt5widgets5 \
  qtbase5-dev \
  libgl1 \
  libglib2.0-0 \
  locales \
  software-properties-common && \
  rm -rf /var/lib/apt/lists/*

# Configure locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
  LC_ALL=en_US.UTF-8

# (Optional) ROS 2 setup, if you need ROS 2 functionality:
ARG ROS_DISTRO=humble
RUN add-apt-repository universe && \
  wget -qO - https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" \
  | tee /etc/apt/sources.list.d/ros2-latest.list > /dev/null && \
  apt-get update && \
  apt-get install -y \
  ros-${ROS_DISTRO}-desktop-full \
  python3-rosdep \
  python3-colcon-common-extensions \
  python3-vcstool && \
  rm -rf /var/lib/apt/lists/* && \
  rosdep init && rosdep update

# Source ROS 2 setup script on shell startup (if using ROS 2)
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# Ensure python3 points to /usr/bin/python3.10 (the ROS Python)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
  && update-alternatives --set python3 /usr/bin/python3.10

# Upgrade pip to the latest version using the ROS Python
RUN /usr/bin/python3.10 -m pip install --upgrade pip

# Install additional Python packages using the ROS Python
RUN /usr/bin/python3.10 -m pip install matplotlib opencv-python imageio

# (Optional) Install PyTorch from pip with CUDA support using the ROS Python.
# Adjust the index URL or version as needed.
RUN /usr/bin/python3.10 -m pip install --break-system-packages --ignore-installed torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN /usr/bin/python3.10 -m pip install "numpy<2.0"

# Copy SAM 2 RT source code into the container
COPY ./ /workspace/sam2_rt

# (Optional) Pre-download checkpoints (if available during build)
WORKDIR /workspace/sam2_rt/checkpoints
RUN chmod +x download_ckpts.sh && ./download_ckpts.sh

# Set working directory for the project
WORKDIR /workspace/sam2_rt

# Install SAM 2 RT in editable mode
RUN pip3 install -e .

# (Removed build_ext step from build time)

# Copy the entrypoint script and configure it
WORKDIR /workspace/sam2_rt/docker
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set the entrypoint and default command
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
