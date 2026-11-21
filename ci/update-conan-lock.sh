#!/usr/bin/env bash
# Regenerates conan.lock. One lockfile covers every platform: the Linux graph (all graphics
# backends enabled) is created first, then the Windows graph is merged into it.
# Bump ci/conan-index.ref together with the lockfile when the index recipes changed.
#
# Not locked (unresolvable or conflicting at the time of writing):
#  - mcnopper-egl (Windows EGL): not in conan-index yet
#  - vulkan-validationlayers (with_vulkan_validation): conflicts with the pinned vulkan-headers
#
# Usage: ci/update-conan-lock.sh   (conan on PATH, remotes configured with ci/conan-setup-remotes.sh)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
rm -f conan.lock

# Revisions do not depend on the compiler, but the build profile has to be able to resolve it
common=(-pr:b=default -s build_type=Release -o '&:with_opengl=True' -o '&:with_vulkan=True' -o '&:build_benchmarks=True')

conan lock create . -pr:h=conan/profiles/linux-gcc "${common[@]}" -o '&:with_egl=True' -o '&:with_webgpu=True' --lockfile-out=conan.lock
conan lock create . -pr:h=conan/profiles/windows-msvc "${common[@]}" -o '&:with_d3d12=True' \
  --lockfile=conan.lock --lockfile-out=conan.lock
