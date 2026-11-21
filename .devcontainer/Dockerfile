# syntax=docker/dockerfile:1.7
ARG UBUNTU_VERSION=24.04
FROM mcr.microsoft.com/devcontainers/base:ubuntu-${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# Create a non-root user (vscode) if the base image didn't already
# (devcontainers/base provides it, but this is safe)
RUN if ! id -u vscode >/dev/null 2>&1; then \
    groupadd -g 1000 vscode && useradd -m -s /bin/bash -u 1000 -g 1000 vscode; \
    fi

# Add the repository for up-to-date mesa drivers
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:kisak/kisak-mesa \
    && rm -rf /var/lib/apt/lists/*

# Core build deps, C++ toolchains, debuggers, and helpers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git curl ca-certificates pkg-config \
    cmake ninja-build \
    gdb lldb \
    # clang-format is 18 on Ubuntu 24.04; .github/workflows/clang-format-check.yml must use the same major version
    clang clang-tidy clang-format lld \
    ccache \
    python3 python3-pip python3-venv \
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
    # OpenGL and EGL development (GLEW comes from Conan; egl/system checks for libegl-dev)
    libegl-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    # Conan's vulkan-loader builds its X11/XCB window-system support from these
    libx11-xcb-dev libxcb-dri3-dev libxcb-present-dev libxcb-randr0-dev libxkbcommon-dev libwayland-dev \
    # Vulkan runtime (lavapipe software driver); headers, loader and glslc come from Conan
    vulkan-tools mesa-vulkan-drivers vulkan-validationlayers \
    # noVNC stack
    xvfb x11vnc openbox supervisor \
    websockify novnc \
    # utils
    xterm dbus-x11 doxygen \
    && rm -rf /var/lib/apt/lists/*

# Conan is installed per workspace in a venv at the version pinned in ci/requirements-conan.txt
# (see postCreateCommand in devcontainer.json).

# Direct3D 12 on Linux: cross-compile with MinGW-w64 (ci/mingw-w64-toolchain.cmake)
# and run the tests under Wine. vkd3d-proton implements D3D12 on Vulkan, DXVK
# provides its DXGI. Wine comes from WineHQ: Ubuntu's Wine 9.0 doesn't expose
# the Vulkan extensions DXVK 3 requires.
ARG VKD3D_PROTON_VERSION=3.1
ARG VKD3D_PROTON_SHA256=b85ea8219b71093a3924913dd7aae6172333dc0c069983ed510816d67f02120b
ARG DXVK_VERSION=3.1.1
ARG DXVK_SHA256=40565b4a724aadc4433fa4e010b4b23916d9b1f1baeee64e17186db94f54e608
RUN dpkg --add-architecture i386 \
    && mkdir -pm755 /etc/apt/keyrings \
    && curl -fsSL https://dl.winehq.org/wine-builds/winehq.key -o /etc/apt/keyrings/winehq-archive.key \
    && curl -fsSL https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources \
         -o /etc/apt/sources.list.d/winehq-noble.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
       g++-mingw-w64-x86-64-posix zstd \
    && apt-get install -y --install-recommends winehq-stable \
    && rm -rf /var/lib/apt/lists/*
RUN cd /tmp \
    && curl -fsSL -o vkd3d-proton.tar.zst https://github.com/HansKristian-Work/vkd3d-proton/releases/download/v${VKD3D_PROTON_VERSION}/vkd3d-proton-${VKD3D_PROTON_VERSION}.tar.zst \
    && echo "${VKD3D_PROTON_SHA256}  vkd3d-proton.tar.zst" | sha256sum -c - \
    && tar --zstd -xf vkd3d-proton.tar.zst -C /opt && mv /opt/vkd3d-proton-${VKD3D_PROTON_VERSION} /opt/vkd3d-proton \
    && curl -fsSL -o dxvk.tar.gz https://github.com/doitsujin/dxvk/releases/download/v${DXVK_VERSION}/dxvk-${DXVK_VERSION}.tar.gz \
    && echo "${DXVK_SHA256}  dxvk.tar.gz" | sha256sum -c - \
    && tar xzf dxvk.tar.gz -C /opt && mv /opt/dxvk-${DXVK_VERSION} /opt/dxvk \
    && rm -f /tmp/vkd3d-proton.tar.zst /tmp/dxvk.tar.gz
ENV VCL_D3D12_WINE_DLLS="/opt/vkd3d-proton/x64/d3d12.dll;/opt/vkd3d-proton/x64/d3d12core.dll;/opt/dxvk/x64/dxgi.dll"

# ccache settings
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
