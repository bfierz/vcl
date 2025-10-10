# syntax=docker/dockerfile:1.7
ARG UBUNTU_VERSION=22.04
FROM mcr.microsoft.com/devcontainers/base:ubuntu-${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG DEFAULT_COMPILER=gcc

# Create a non-root user (vscode) if the base image didn't already
# (devcontainers/base provides it, but this is safe)
RUN if ! id -u vscode >/dev/null 2>&1; then \
    groupadd -g 1000 vscode && useradd -m -s /bin/bash -u 1000 -g 1000 vscode; \
    fi

# Add the repository for up-to-date mesa drivers
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:kisak/kisak-mesa 

# Core build deps, C++ toolchains, debuggers, and helpers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git curl ca-certificates pkg-config \
    cmake ninja-build \
    gdb lldb \
    clang clang-tidy clang-format lld \
    ccache \
    python3 python3-pip python3-venv pipx \
    # OpenGL & X11 runtime bits for Qt GUI
    libgl1-mesa-dri libgl1-mesa-dev libglu1-mesa-dev \
    libxkbcommon-x11-0 libx11-xcb1 libxcb1 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render0 libxcb-render-util0 \
    libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcb-xkb1 \
    x11-apps \
    # Qt 6 development (widgets, tools, qml)
    qt6-base-dev qt6-base-dev-tools \
    qt6-tools-dev qt6-tools-dev-tools \
    qt6-declarative-dev \
    # OpenGL development
    libglew-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    # noVNC stack
    xvfb x11vnc openbox supervisor \
    websockify novnc \
    # utils
    xterm x11-apps dbus-x11 doxygen \
    && rm -rf /var/lib/apt/lists/*

# Conan 2 (system-wide via pipx; keeps it isolated & on PATH)
ENV PIPX_HOME=/opt/pipx
ENV PIPX_BIN_DIR=/usr/local/bin
ENV PIPX_MAN_DIR=/usr/local/share/man
RUN pipx ensurepath \
    && pipx install "conan>=2.0,<3.0" \
    && conan --version

# Default compiler selection (gcc or clang)
ENV CCACHE_DIR=/usr/local/.ccache
ENV CCACHE_MAXSIZE=10G
ENV CCACHE_COMPRESS=1

# Qt tools on PATH
ENV PATH="/usr/lib/qt6/bin:${PATH}"

# noVNC path hints (Ubuntu novnc package installs here)
ENV NOVNC_DIR=/usr/share/novnc
ENV WEBSOCKIFY_CMD=/usr/bin/websockify

# Make sure the vscode user owns the ccache dir
RUN if ! id -u vscode >/dev/null 2>&1; then groupadd -g 1000 vscode && useradd -m -s /bin/bash -u 1000 -g 1000 vscode; fi
RUN mkdir -p /usr/local/.ccache && chown -R vscode:vscode /usr/local/.ccache

# Bootstrap scripts
COPY scripts /tmp/dev-scripts
RUN install -m 0755 -o root -g root /tmp/dev-scripts/start-vnc.sh /usr/local/bin/start-vnc.sh \
    && install -m 0644 -o root -g root /tmp/dev-scripts/supervisord.conf /etc/supervisor/conf.d/novnc.conf \
    && rm -rf /tmp/dev-scripts

USER vscode
WORKDIR /workspaces
