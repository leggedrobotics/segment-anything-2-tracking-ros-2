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

# If we are running as root, set up sudoers
if [ "$EUID" -eq 0 ]; then
  # Enable sudo access without password
  echo "root ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
  echo "$HOST_USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

  # Now switch to $HOST_USERNAME and run the container's CMD
  exec sudo -E -u "$HOST_USERNAME" --preserve-env=HOME \
       bash -c "cd $HOME && exec \"$@\""
else
  # We’re already a non-root user; just run the CMD
  exec "$@"
fi
