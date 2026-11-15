#!/bin/sh
# Runs a D3D12 test binary (MinGW build) under Wine on Linux.
#
# vkd3d-proton implements D3D12 on Vulkan; DXVK provides the DXGI it needs for
# swap chains (Wine's own DXGI only works with Wine's built-in D3D12). The DLLs
# must be next to the executable (CMake copies VCL_D3D12_WINE_DLLS there).
#
# Variables already set in the environment take precedence.

# Prefer the DLLs next to the executable; skip the Mono/Gecko install prompts
# when Wine creates a new prefix
export WINEDLLOVERRIDES="${WINEDLLOVERRIDES:-d3d12,d3d12core,dxgi=n;mscoree,mshtml=}"
export WINEDEBUG="${WINEDEBUG:--all}"
export VKD3D_DEBUG="${VKD3D_DEBUG:-warn}"
export DXVK_LOG_LEVEL="${DXVK_LOG_LEVEL:-warn}"

# Wine provides no WARP adapter
export VCL_D3D12_DEVICE="${VCL_D3D12_DEVICE:-hardware}"

# lavapipe lacks tiled resources tier 2 (feature level 12_0) and FP32 denorm
# controls (shader model 6.2+); neither is used by the tests
export VKD3D_FEATURE_LEVEL="${VKD3D_FEATURE_LEVEL:-12_1}"
export VKD3D_SHADER_MODEL="${VKD3D_SHADER_MODEL:-6_6}"

exec wine "$@"
