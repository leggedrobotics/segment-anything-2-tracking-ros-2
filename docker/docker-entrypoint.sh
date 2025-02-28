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

# If running as root, set up sudoers and switch to the host user while sourcing ROS Humble
if [ "$EUID" -eq 0 ]; then
  # Enable sudo access without a password
  echo "root ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
  echo "$HOST_USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

  # Switch to $HOST_USERNAME:
  #  - Source ROS Humble
  #  - Change directory to the desired workspace location
  #  - Execute the Python node
  exec sudo -E -u "$HOST_USERNAME" --preserve-env=HOME \
       bash -c 'source /opt/ros/humble/setup.bash && \
                cd "$HOME/boulder_ws/src/segment-anything-2-real-time-ros-2" && \
                python3 ros2/object_tracker_node.py'
else
  # If already non-root, source ROS Humble,
  # change to the desired workspace location, and execute the Python node.
  source /opt/ros/humble/setup.bash
  cd "$HOME/boulder_ws/src/segement-anything-2-real-time-ros-2"
  exec python3 ros2/object_tracker_node.py
fi
