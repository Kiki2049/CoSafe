# Bridge container for CoLMDriver multi-ego Autoware integration.
# Aligned with TUM autoware_carla_leaderboard verified stack:
#   - Same base image: ros:humble-ros-base-jammy
#   - Same apt packages (subset relevant to bridge)
#   - Same autoware_*_msgs repos + exact versions from TUM's docker/ros2.repos
#   - Same Python deps versions from TUM's docker/requirements.txt
#   - Same colcon build strategy as TUM README Step 4
#
# This container does NOT contain Autoware Universe itself — it only provides
# the ROS2 message bindings so autoware_bridge.py can publish/subscribe the
# topics that a separate Autoware Universe container uses.

FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive

# --- apt packages: aligned with TUM Dockerfile lines 8-37 ---
# We keep the same set minus media libs (ffmpeg/cairo/jpeg/tiff) that only
# their leaderboard visualization needs.
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
        build-essential \
        curl \
        git \
        pkg-config \
        python3-colcon-common-extensions \
        python3-dev \
        python3-pip \
        python3-rosdep \
        python3-vcstool \
        ros-"${ROS_DISTRO}"-lanelet2 \
        ros-"${ROS_DISTRO}"-launch \
        ros-"${ROS_DISTRO}"-launch-ros \
        ros-"${ROS_DISTRO}"-rmw-cyclonedds-cpp \
        sudo \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps: from TUM's docker/requirements.txt (bridge-relevant subset) ---
RUN python3 -m pip install --upgrade pip wheel \
    && python3 -m pip install "setuptools<=70.3.0" \
    && python3 -m pip install --no-cache-dir \
        numpy~=1.26.4 \
        scipy~=1.14.1 \
        transforms3d~=0.4.1 \
        pexpect~=4.9.0 \
        flask~=3.0.0

# --- autoware msg packages: exact repos + versions from TUM's docker/ros2.repos ---
ENV BRIDGE_WS=/bridge_ws
WORKDIR ${BRIDGE_WS}

# Reproduce TUM's ros2.repos exactly (4 msg packages, pinned versions)
RUN mkdir -p src/external && \
    cd src/external && \
    git clone --depth 1 -b 1.4.0 \
        https://github.com/autowarefoundation/autoware_msgs.git && \
    git clone --depth 1 -b v0.41.0 \
        https://github.com/tier4/tier4_autoware_msgs.git && \
    git clone \
        https://github.com/autowarefoundation/autoware_adapi_msgs.git && \
    cd autoware_adapi_msgs && \
    git checkout e62ea6fdb9ee50e49c7642c77e85250cb15b8e80 && \
    cd .. && \
    git clone -b 1.8.0 \
        https://github.com/autowarefoundation/autoware_internal_msgs.git

# --- colcon build: same as TUM README Step 4 ---
RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-geographic-msgs
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi \
    && rosdep update \
    && apt-get update \
    && rosdep install -y --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} || true \
    && /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
       colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- ROS environment setup: aligned with TUM Dockerfile lines 71-82 ---
ENV ROS_PATH=/opt/ros/"${ROS_DISTRO}"/lib/python3.10/site-packages
ENV PYTHONPATH="${PYTHONPATH}":"${ROS_PATH}"
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV PYTHONUNBUFFERED=1

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ${BRIDGE_WS}/install/setup.bash" >> ~/.bashrc

# --- Our bridge code (bind-mounted at runtime; COPY as fallback) ---
COPY bridge_converter.py autoware_bridge.py /opt/bridge/
ENV PYTHONPATH="/opt/bridge:${PYTHONPATH}"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "/opt/bridge/autoware_bridge.py", "--host", "0.0.0.0", "--port", "8100"]
