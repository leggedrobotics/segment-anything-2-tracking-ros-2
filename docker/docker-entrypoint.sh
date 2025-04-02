#!/bin/bash
set -e

# Optional: Change to the workspace directory if needed
cd /workspace/sam2_rt
# Check if the extension has already been built (using a marker file)
if [ ! -f extension_built.marker ]; then
  echo "Building C++/CUDA extensions..."
  python3 setup.py build_ext --inplace
  touch extension_built.marker
else
  echo "C++/CUDA extensions already built, skipping rebuild."
fi

# Set home for host user
export HOME=/home/$HOST_USERNAME
export USER=$HOST_USERNAME

if [ "$EUID" -eq 0 ]; then
  # Enable sudo access without a password
  echo "root ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
  echo "$HOST_USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

  # Switch to $HOST_USERNAME, source ROS, change to the ros2 directory,
  # run the Python node, and finally launch an interactive shell.
  sudo -E -u "$HOST_USERNAME" --preserve-env=HOME bash -c '\
    source /opt/ros/humble/setup.bash && \
    cd "$HOME/boulder_ws/src/segment-anything-2-real-time-ros-2/ros2" && \
    python3 object_tracker_node.py; \
    exec bash'
else
  # For non-root users, source ROS, change directory, run the Python node,
  # then drop into an interactive shell.
  source /opt/ros/humble/setup.bash
  cd "$HOME/boulder_ws/src/segment-anything-2-real-time-ros-2/ros2"
  python3 object_tracker_node.py
  exec bash
fi
