# world_embeddings: ROS 2 Humble + ONNX Runtime + colcon build
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base-jammy AS builder

ENV ONNXRUNTIME_VERSION=1.22.0
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime

# Build and runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libeigen3-dev \
    libopencv-dev \
    ros-${ROS_DISTRO}-cv-bridge \
    python3-colcon-common-extensions \
    python3-rosdep \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime C++ (prebuilt: x64 for amd64, aarch64 for arm64 / Apple Silicon)
COPY docker/install_onnxruntime.sh /tmp/install_onnxruntime.sh
RUN chmod +x /tmp/install_onnxruntime.sh \
    && /tmp/install_onnxruntime.sh "${ONNXRUNTIME_VERSION}" "${ONNXRUNTIME_ROOT}"

WORKDIR /ros_ws/src
COPY . world_embeddings

WORKDIR /ros_ws
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && colcon build --packages-select world_embeddings \
    --cmake-args -DONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}

# Runtime image
FROM ros:${ROS_DISTRO}-ros-base-jammy

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-cv-bridge \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder ${ONNXRUNTIME_ROOT} ${ONNXRUNTIME_ROOT}
COPY --from=builder /ros_ws/install /ros_ws/install
COPY --from=builder /ros_ws/src /ros_ws/src

WORKDIR /ros_ws
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /ros_ws/env.sh \
    && echo "source /ros_ws/install/setup.bash" >> /ros_ws/env.sh \
    && echo "export LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT}/lib:\${LD_LIBRARY_PATH}" >> /ros_ws/env.sh

ENTRYPOINT ["/bin/bash", "-c", "source /ros_ws/env.sh && exec \"$@\"", "--"]
CMD ["ros2", "launch", "world_embeddings", "world_embeddings.launch.py"]
