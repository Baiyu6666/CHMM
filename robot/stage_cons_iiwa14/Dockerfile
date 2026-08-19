# syntax=docker/dockerfile:1

FROM ros:noetic-ros-base

ARG USERNAME=ros
ARG USER_UID=1000
ARG KUKA_FRI_REPOSITORY=git@github.com:epfl-lasa/kuka_fri.git
ARG KUKA_FRI_REF=59af82a8524b37e08a1424a50605e7fe70f11ec9

SHELL ["/bin/bash", "-c"]

RUN useradd --uid "${USER_UID}" -m -s /bin/bash "${USERNAME}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       curl \
       git \
       iproute2 \
       iputils-ping \
       iptables \
       libeigen3-dev \
       liborocos-kdl-dev \
       python-is-python3 \
       ros-noetic-controller-interface \
       ros-noetic-controller-manager \
       ros-noetic-control-toolbox \
       ros-noetic-geometry-msgs \
       ros-noetic-hardware-interface \
       ros-noetic-joint-state-controller \
       ros-noetic-joint-limits-interface \
       ros-noetic-kdl-parser \
       ros-noetic-nav-msgs \
       ros-noetic-pluginlib \
       ros-noetic-rosbag \
       ros-noetic-realtime-tools \
       ros-noetic-robot-state-publisher \
       ros-noetic-roscpp \
       ros-noetic-sensor-msgs \
       ros-noetic-std-msgs \
       ros-noetic-std-srvs \
       ros-noetic-tf \
       ros-noetic-tf2-ros \
       ros-noetic-urdf \
       ros-noetic-vrpn-client-ros \
       ros-noetic-xacro \
    && rm -rf /var/lib/apt/lists/*

# The FRI client is not vendored here. BuildKit forwards the user's SSH agent
# only for this layer, so private-repository credentials are not stored.
RUN mkdir -p -m 0700 /root/.ssh \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
RUN --mount=type=ssh \
    git clone "${KUKA_FRI_REPOSITORY}" /tmp/kuka_fri \
    && if [[ -n "${KUKA_FRI_REF}" ]]; then git -C /tmp/kuka_fri checkout "${KUKA_FRI_REF}"; fi \
    && curl -fsSL \
       https://gist.githubusercontent.com/matthias-mayr/0f947982474c1865aab825bd084e7a92/raw/244f1193bd30051ae625c8f29ed241855a59ee38/0001-Config-Disables-SIMD-march-native-by-default.patch \
       -o /tmp/kuka_fri/no-native.patch \
    && git -C /tmp/kuka_fri apply /tmp/kuka_fri/no-native.patch \
    && cd /tmp/kuka_fri \
    && ./waf configure \
    && ./waf \
    && ./waf install \
    && rm -rf /tmp/kuka_fri \
    && ldconfig

RUN mkdir -p /home/${USERNAME}/ros_ws/src \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/ros_ws
COPY --chown=${USERNAME}:${USERNAME} ros_ws/src/ /home/${USERNAME}/ros_ws/src/

USER ${USERNAME}
WORKDIR /home/${USERNAME}/ros_ws
RUN source /opt/ros/noetic/setup.bash \
    && catkin_make -DCMAKE_BUILD_TYPE=RelWithDebInfo

# docker compose exec does not pass through ENTRYPOINT. Export the generated
# workspace environment here as well so direct rosnode/rosservice commands work.
ENV PATH="/home/${USERNAME}/ros_ws/devel/bin:/opt/ros/noetic/bin:${PATH}" \
    ROS_PACKAGE_PATH="/home/${USERNAME}/ros_ws/src:/opt/ros/noetic/share" \
    PYTHONPATH="/home/${USERNAME}/ros_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages" \
    LD_LIBRARY_PATH="/home/${USERNAME}/ros_ws/devel/lib:/opt/ros/noetic/lib" \
    CMAKE_PREFIX_PATH="/home/${USERNAME}/ros_ws/devel:/opt/ros/noetic"

RUN printf '%s\n' \
      'source /opt/ros/noetic/setup.bash' \
      'source /home/ros/ros_ws/devel/setup.bash' \
      >> /home/${USERNAME}/.bashrc

COPY --chown=${USERNAME}:${USERNAME} docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["roscore"]
