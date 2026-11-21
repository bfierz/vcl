#!/usr/bin/env bash
# Regenerates conan.lock. One lockfile covers every platform: the Linux graph (all graphics
# backends, the benchmarks and the examples) is created first, then the further graphs are merged
# into it, so that every package conanfile.py can request is locked.
# Bump ci/conan-index.ref together with the lockfile when the index recipes changed.
#
# Graphs:
#  - Linux, C++17: OpenGL, EGL, Vulkan with validation layers, benchmarks, examples
#  - Linux, C++20: adds with_webgpu; Dawn needs C++20, the C++17 profiles report it invalid
#  - Windows: OpenGL/Vulkan with the examples, and Direct3D 12 (brings the dxc tool_requires)
#
# Not locked (unresolvable at the time of writing):
#  - mcnopper-egl (Windows EGL): not in conan-index yet
#  - imguizmo/cci.20261010 (Direct3D 12 examples on Windows): not in the pinned conan-index yet,
#    so the Direct3D 12 graph is locked without the examples
#
# Usage: ci/update-conan-lock.sh   (conan on PATH, remotes configured with ci/conan-setup-remotes.sh)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
rm -f conan.lock

# Revisions do not depend on the compiler, but the build profile has to be able to resolve it
common=(-pr:b=default -s build_type=Release -o '&:with_opengl=True' -o '&:with_vulkan=True' -o '&:build_benchmarks=True' -o '&:build_examples=True')

lock() { conan lock create . "$@" --lockfile-out=conan.lock; }

lock -pr:h=conan/profiles/linux-gcc "${common[@]}" -o '&:with_egl=True' -o '&:with_vulkan_validation=True'
lock -pr:h=conan/profiles/linux-gcc "${common[@]}" -o '&:with_webgpu=True' -s compiler.cppstd=20 --lockfile=conan.lock
lock -pr:h=conan/profiles/windows-msvc "${common[@]}" -o '&:with_d3d12=False' --lockfile=conan.lock
# Direct3D 12 without the examples: its only example dependency (imguizmo) is not resolvable yet
lock -pr:h=conan/profiles/windows-msvc "${common[@]}" -o '&:with_d3d12=True' -o '&:build_examples=False' --lockfile=conan.lock
