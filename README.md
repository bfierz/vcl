VCL
===

[![Build Status](https://dev.azure.com/basilfierz/VCL/_apis/build/status/vcl?branchName=master)](https://dev.azure.com/basilfierz/VCL/_build/latest?definitionId=2&branchName=master)
[![codecov](https://codecov.io/gh/bfierz/vcl/branch/master/graph/badge.svg)](https://codecov.io/gh/bfierz/vcl)

The Visual Computing Library (VCL) started as a repository for algorithms and data structures developed during my PhD student time. As the name says it was used to visual computing techniques, which in this case were visual simulations.

Requirements
------------

Requires a C++17 capable compiler, CMake 3.23 or newer, and Python 3 for the Conan client.

Building
--------

Third-party dependencies come from Conan 2 (see [docs/adr/0001-conan-with-find-package-seam.md](docs/adr/0001-conan-with-find-package-seam.md)); there are no submodules.
The Conan client is pinned in `ci/requirements-conan.txt`. The build uses explicit `conan install` followed by CMake presets:

```sh
python3 -m venv .venv-conan && .venv-conan/bin/pip install -r ci/requirements-conan.txt
source .venv-conan/bin/activate        # Windows: .venv-conan\Scripts\activate
conan profile detect --force           # creates the build profile ("default")
ci/conan-setup-remotes.sh              # Windows: ci/conan-setup-remotes.ps1
conan install . -pr:h=conan/profiles/linux-gcc -pr:b=default -s build_type=Release --build=missing
cmake --preset conan-release           # Windows (Visual Studio): cmake --preset conan-default
cmake --build --preset conan-release   # Windows: add --config Release
```

* Host profiles live in `conan/profiles`: `linux-gcc`, `linux-aarch64-gcc`, `linux-arm-gcc`, `macos-apple-clang`, `windows-msvc`, `mingw-w64` and `emscripten`.
  The profile's `compiler.cppstd` is the only source of the C++ standard.
* `ci/conan-setup-remotes.sh` registers `conancenter` first, then the recipes from [bfierz/conan-index](https://github.com/bfierz/conan-index): through the `b4z` remote when `VCL_CONAN_B4Z_URL` and a read-only token are set, otherwise from a clone pinned in `ci/conan-index.ref`.
* `conan.lock` is committed and picked up by `conan install`. `ci/update-conan-lock.sh` regenerates it.
* Options are set with `-o "&:<name>=True"` (or in the profile) and select the `VCL_*` CMake options:

| Conan option             | Default | CMake option             | Notes                                                       |
|--------------------------|---------|--------------------------|-------------------------------------------------------------|
| `with_opengl`            | False   | `VCL_OPENGL_SUPPORT`     |                                                             |
| `with_egl`               | False   | `VCL_EGL_SUPPORT`        | Needs `with_opengl`                                         |
| `with_vulkan`            | False   | `VCL_VULKAN_SUPPORT`     |                                                             |
| `with_vulkan_validation` | False   |                          | Packages the validation layers, used at runtime only        |
| `with_d3d12`             | False   | `VCL_D3D12_SUPPORT`      | Windows                                                     |
| `with_webgpu`            | False   | `VCL_WEBGPU_SUPPORT`     | Resolves Dawn; the backend is not built until the API port |
| `build_tests`            | True    | `VCL_BUILD_TESTS`        |                                                             |
| `build_benchmarks`       | True    | `VCL_BUILD_BENCHMARKS`   |                                                             |
| `build_examples`         | False   | `VCL_BUILD_EXAMPLES`     |                                                             |
| `build_tools`            | True    | `VCL_BUILD_TOOLS`        |                                                             |

Dawn needs C++20 to build. VCL only uses its C API, so the profiles keep C++17 and build Dawn alone with C++20 (`dawn/*:compiler.cppstd=20`). A profile of your own needs the same line for `with_webgpu`.

C++ feature matrices:
A good overview on the feature support for different compiler versions can be found here: https://en.cppreference.com/w/cpp/compiler_support

Detailed reports at the vendor's pages:

* MSVC: https://msdn.microsoft.com/en-us/library/hh567368.aspx, https://docs.microsoft.com/en-us/cpp/visual-cpp-language-conformance?view=vs-2017
* Intel: https://software.intel.com/en-us/articles/c14-features-supported-by-intel-c-compiler
* Clang: https://clang.llvm.org/cxx_status.html
* GCC: https://gcc.gnu.org/projects/cxx-status.html

Supported compilers:

* MSVC >= 2017
* Clang >= 3.9
* GCC >= 5.1
* Intel ICC >= 17

Compatibility
-------------

VCL generally supports the CPU architectures x86, x86-64, and ARM. Test coverage for ARM is still in development.
As host platforms, VCL supports Windows, Linux and MacOS.

External Libraries
------------------

VCL depends on a number of external libraries. Most are resolved by Conan through `find_package`; the versions below are those pinned in `conanfile.py` and `conan.lock`. A few small libraries are kept in the source tree.

| Library                       | License    | Version        | Source                                       | Notes                                    |
|-------------------------------|------------|----------------|----------------------------------------------|------------------------------------------|
| Abseil Common Libraries (C++) | Apache-2.0 | 20250814.2     | https://abseil.io                            |                                          |
| Eigen 3                       | MPL2       | 3.4.1          | https://eigen.tuxfamily.org                  |                                          |
| {fmt}                         | MIT        | 11.2.0         | https://github.com/fmtlib/fmt                |                                          |
| GLEW                          | BSD/MIT    | 2.2.0          | https://github.com/nigels-com/glew           | `with_opengl`                            |
| EGL                           |            | system / mcnopper-egl | | `with_egl`: `egl/system` on Linux, `mcnopper-egl` (conan-index) on Windows |
| Vulkan Headers and Loader     | Apache-2.0 | 1.4.313.0      | https://github.com/KhronosGroup/Vulkan-Headers | `with_vulkan`                          |
| Vulkan Validation Layers      | Apache-2.0 | 1.4.350.0      | https://github.com/KhronosGroup/Vulkan-ValidationLayers | `with_vulkan_validation`    |
| SPIRV-Cross                   | Apache-2.0 | cci.20211113   | https://github.com/KhronosGroup/SPIRV-Cross  | `with_vulkan`                            |
| DirectX-Headers               | MIT        | 1.614.0        | https://github.com/microsoft/DirectX-Headers | `with_d3d12`                             |
| Dawn                          | Apache-2.0 | 159.0.8154     | https://dawn.googlesource.com/dawn           | `with_webgpu`, resolved only             |
| shaderc (glslc)               | Apache-2.0 | 2026.4         | https://github.com/google/shaderc            | Build tool, OpenGL and Vulkan shaders    |
| DirectX Shader Compiler (dxc) | NCSA       | 1.9.2609       | https://github.com/microsoft/DirectXShaderCompiler | Build tool, `with_d3d12` (conan-index) |
| Google Test                   | BSD-3      | 1.17.0         | https://github.com/google/googletest         | `build_tests`                            |
| JSON for Modern C++           | MIT        | 3.12.0         | https://github.com/nlohmann/json             | `build_tests`                            |
| Google Benchmark              | Apache-2.0 | 1.9.5          | https://github.com/google/benchmark          | `build_benchmarks`                       |
| foonathan/memory              | Zlib       | 0.7.3          | https://github.com/foonathan/memory          | `build_benchmarks`                       |
| Geometric Tools Engine        | BSL-1.0    | 8.0            | https://github.com/davideberly/GeometricTools | `build_benchmarks`                     |
| Dear ImGui                    | MIT        | 1.92.9b        | https://github.com/ocornut/imgui             | `build_examples`                         |
| GLFW                          | Zlib       | 3.4            | https://www.glfw.org                         | `build_examples`                         |
| ImPlot                        | MIT        | 1.1            | https://github.com/epezent/implot            | `build_examples`, OpenGL                 |
| ImFileDialog                  | MIT        | cci.20240916   | https://github.com/dfranx/ImFileDialog       | `build_examples`, OpenGL (conan-index)   |
| ImGuizmo                      | MIT        | cci.20261010   | https://github.com/CedricGuillemet/ImGuizmo  | `build_examples`, D3D12 (conan-index)    |
| C++ commandline parsing       | MIT        | 3.0.0          | https://github.com/jarro2783/cxxopts         | In source tree                           |
| Expected lite                 | BSL-1.0    | 0.6.2          | https://github.com/martinmoene/expected-lite | In source tree                           |
| Tet-tet intersection          | Custom     |                | https://github.com/erich666/jgt-code/blob/master/Volume_07/Number_2/Ganovelli2002/tet_a_tet.h | Refactored implementation. The license is included [here](./doc/license_ganovelli_tet_a_tet.md) |

The SIMD module uses support library to provide trigonometry, logarithm and power functions. The SSE and NEON implementations are taken directly from [Julien Pommier's homepage](http://gruntthepeon.free.fr/ssemath/). The AVX version is thanks to Giovanni Garberoglio and the AVX512 to [JishinMaster](https://github.com/JishinMaster/simd_utils).
