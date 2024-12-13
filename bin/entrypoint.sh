#!/bin/bash

# Stop the script if any command fails
set -e

# Debugging: Print the executed commands (optional)
# Uncomment this line if you want to see all commands being executed
# set -x
# Debugging environment variables

# echo "HOST_USERNAME=$HOST_USERNAME"
echo "HOST_UID=$HOST_UID"
echo "HOST_GID=$HOST_GID"

# Use the ROS_DISTRO environment variable
if [ -z "$ROS_DISTRO" ]; then
    echo "ERROR: ROS_DISTRO is not set!"
    exit 1
fi

# Source the ROS setup file
if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    source "/opt/ros/$ROS_DISTRO/setup.bash"
else
    echo "ERROR: Could not find setup.bash for ROS_DISTRO=$ROS_DISTRO."
    exit 1
fi

echo "Exiting entrypoint"

