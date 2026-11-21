# 1. Dependencies through Conan 2, with `find_package` as the only seam

Date: 2026-11-19

## Status

Accepted

## Context

VCL gets its dependencies in four ways: git submodules, FetchContent, CPM, and headers copied into the tree. The CMake files that use a dependency also decide where it comes from. The `vcl_ext_*` variables were once a seam with two adapters (submodules and Conan 1). Commit c180884 removed the Conan adapter, so only the submodule adapter remains, and targets such as `absl_variant` leak past it. The CI matrix built both C++14 and C++17.

## Decision

- Dependencies come from Conan 2: ConanCenter first, then `bfierz/conan-index` for the recipes ConanCenter doesn't have. The index is reached either through the b4z remote or through a pinned `local-recipes-index` clone.
- VCL's CMake files only call `find_package(... CONFIG REQUIRED)` and link the namespaced targets. They never know where a dependency came from. `vcl_ext_*`, `vcl_copy_target`, `vcl_combine_targets` and the subproject overrides are deleted.
  - Where a package's upstream config exports no namespaced target, VCL links the name upstream exports. glfw exports only `glfw`.
  - What `find_package` can't provide, such as the folder of imgui's backend sources, is a plain CMake cache variable (`VCL_IMGUI_BACKEND_DIR`). `conanfile.py` sets it in `generate()`; other builds pass it with `-D`.
- The build runs `conan install` explicitly (`CMakeDeps` + `CMakeToolchain`), then `cmake --preset conan-*`. The cmake-conan dependency provider is not used.
- Consumers get VCL through its installed `vclConfig.cmake` (`vcl::` targets) or as a Conan requirement. `add_subdirectory(vcl)` works only if the parent project provides the same `find_package` targets.
- Only driver-bound libraries stay on the system: OpenGL, CUDA and OpenCL, and EGL on Linux (through ConanCenter's `egl/system`). Vulkan isn't one of them: VCL links only the vendor-neutral headers and loader, and the loader finds the vendor driver at runtime. So `vulkan-headers` and `vulkan-loader` come from ConanCenter and no Vulkan SDK is needed. EGL on Windows comes from a `mcnopper-egl` recipe in conan-index.
- C++17 is the minimum standard. The Conan profile's `compiler.cppstd` is the only source of truth for the standard.

## Alternatives considered

- **cmake-conan provider:** it would keep a plain `cmake -B build`. It needs CMake 3.24, isn't considered stable, and its README says to run Conan separately for toolchain-level settings, which the ARM, aarch64 and MinGW cross builds need.
- **Keep `vcl_ext_*` as override points for embedders:** this would keep a second adapter that nobody tests. `find_package` already lets a parent project supply the targets.
- **Keep C++14:** this would pin abseil below 20250512 and googletest below 1.17, and double the binaries to build and cache, because the abseil package ID includes `cppstd`.

## Consequences

- One way to get dependencies. Versions live in `conanfile.py` and `conan.lock`.
- Building VCL now needs a Conan client: the exact version pinned in `ci/requirements-conan.txt`, installed in a venv.
- Embedding VCL without Conan now means supplying its dependency targets yourself. Zero-setup `add_subdirectory` is gone.
- VCL needs `install()` rules and an exported config package, which it didn't have before.
