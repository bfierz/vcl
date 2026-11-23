#!/usr/bin/env bash
# Regenerates conan.lock. One lockfile covers every platform: the Linux graph (all graphics
# backends, the benchmarks and the examples) is created first, then the further graphs are merged
# into it, so that every package conanfile.py can request is locked.
# Bump ci/conan-index.ref together with the lockfile when the index recipes changed.
#
# Graphs:
#  - Linux, C++17: OpenGL, EGL, Vulkan with validation layers, benchmarks, examples
#  - Linux: adds with_webgpu (the profiles build Dawn alone with C++20)
#  - Windows: OpenGL/Vulkan with the examples, OpenGL with EGL (brings mcnopper-egl), and
#    Direct3D 12 with the examples (brings imguizmo and the dxc tool_requires)
#
# Recipes served by the conan-index are locked by version only. Their revision is a hash that the
# local-recipes-index clone and the recipe uploaded to b4z compute differently, so one revision
# cannot be valid for both. ConanCenter recipes keep their revision.
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
lock -pr:h=conan/profiles/linux-gcc "${common[@]}" -o '&:with_webgpu=True' --lockfile=conan.lock
lock -pr:h=conan/profiles/windows-msvc "${common[@]}" -o '&:with_d3d12=False' --lockfile=conan.lock
lock -pr:h=conan/profiles/windows-msvc "${common[@]}" -o '&:with_d3d12=False' -o '&:with_egl=True' --lockfile=conan.lock
lock -pr:h=conan/profiles/windows-msvc "${common[@]}" -o '&:with_d3d12=True' --lockfile=conan.lock

# Drop "#revision%timestamp" from the packages the conan-index provides
index_dir="${VCL_CONAN_INDEX_DIR:-${root}/.conan-index}"
python3 - "${index_dir}/recipes" <<'PY'
import json, os, re, sys

index_recipes = set(os.listdir(sys.argv[1]))
with open("conan.lock") as f:
    lock = json.load(f)

def strip(ref):
    name = ref.split("/", 1)[0]
    return re.sub(r"#.*$", "", ref) if name in index_recipes else ref

for key in ("requires", "build_requires", "python_requires"):
    if key in lock:
        lock[key] = [strip(r) for r in lock[key]]

with open("conan.lock", "w") as f:
    json.dump(lock, f, indent=4)
    f.write("\n")
PY
