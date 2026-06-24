#!/usr/bin/env bash
set -e

# Aligned with TUM's docker/entrypoint.sh
source /opt/ros/${ROS_DISTRO}/setup.bash

if [ -f "/bridge_ws/install/setup.bash" ]; then
    source /bridge_ws/install/setup.bash
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

exec "$@"
