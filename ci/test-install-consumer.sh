#!/usr/bin/env bash
# Installs a configured VCL build tree and builds ci/install-consumer against it.
#
# Usage: ci/test-install-consumer.sh [build-dir] [build-type]
#   build-dir   the build folder `cmake --preset conan-<type>` created (default: build/Release)
#   build-type  Release or Debug (default: Release)
#
# Dependencies are found through the build folder's Conan toolchain, exactly as a Conan
# consumer would, so the installed vclConfig.cmake has to find_dependency them itself.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(cd "${1:-${root}/build/Release}" && pwd)"
build_type="${2:-Release}"
work="${build_dir}/install-consumer"
prefix="${work}/prefix"

rm -rf "${work}"
cmake --build "${build_dir}"
cmake --install "${build_dir}" --prefix "${prefix}" --config "${build_type}"

toolchain="$(find "${build_dir}" -name conan_toolchain.cmake -print -quit)"
cmake -S "${root}/ci/install-consumer" -B "${work}/build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="${toolchain}" \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_PREFIX_PATH="${prefix}"
cmake --build "${work}/build"
"${work}/build/consumer"
