#!/bin/bash
set -e

# Optional: Change to the workspace directory if needed
cd /workspace/sam2_rt

# Check if the extension has already been built (using a marker file)
if [ ! -f extension_built.marker ]; then
  echo "Building C++/CUDA extensions..."
  python3 setup.py build_ext --inplace
  # Create a marker file to indicate that the build has been done
  touch extension_built.marker
else
  echo "C++/CUDA extensions already built, skipping rebuild."
fi

# Execute the container's CMD
exec "$@"
