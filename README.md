VCL
===

[![Build Status](https://dev.azure.com/basilfierz/VCL/_apis/build/status/vcl?branchName=master)](https://dev.azure.com/basilfierz/VCL/_build/latest?definitionId=2&branchName=master)
[![codecov](https://codecov.io/gh/bfierz/vcl/branch/master/graph/badge.svg)](https://codecov.io/gh/bfierz/vcl)

The Visual Computing Library (VCL) started as a repository for algorithms and data structures developed during my PhD student time. As the name says it was used to visual computing techniques, which in this case were visual simulations.

Requirements
------------

Requires a C++17 capable compiler and CMake 3.23 or newer.

Building
--------

The build uses a pinned Conan 2 client (`ci/requirements-conan.txt`) and CMake presets:

```sh
python3 -m venv .venv-conan && .venv-conan/bin/pip install -r ci/requirements-conan.txt
source .venv-conan/bin/activate        # Windows: .venv-conan\Scripts\activate
conan profile detect --force
ci/conan-setup-remotes.sh              # Windows: ci/conan-setup-remotes.ps1
git submodule update --init --recursive
conan install . -pr:h=conan/profiles/linux-gcc -pr:b=default -s build_type=Release --build=missing
cmake --preset conan-release           # Windows (Visual Studio): cmake --preset conan-default
cmake --build --preset conan-release   # Windows: add --config Release
```

Conan options (`-o "&:with_opengl=True"`, `build_tests`, ...) select the `VCL_*` CMake options.
Use `conan/profiles/windows-msvc` on Windows; `compiler.cppstd` in the profile selects the C++ standard.

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

Packed External Libraries
--------------------------

VCL depends on a number of external libraries. Some are placed directly into the source tree.
Most, however, are managed as linked submodule.

| Library                       | License    | Version  | Source                                       | Notes                     |
|-------------------------------|------------|----------|----------------------------------------------|---------------------------|
| Abseil Common Libraries (C++) | Apache-2.0 | 20211102 | https://abseil.io                            | Release 20211102          |
| C++ commandline parsing       | MIT        | 3.0.0    | https://github.com/jarro2783/cxxopts         | In source tree            |
| Eigen 3                       | MPL2       | 5.0.1    | https://eigen.tuxfamily.org                  |                           |
| Expected lite                 | BSL-1.0    | 0.6.2    | https://github.com/martinmoene/expected-lite | In source tree            |
| Google Test                   | BSD-3      | 1.11.0   | https://github.com/google/googletest         |                           |
| Google Benchmark              | Apache-2.0 | 1.5.5    | https://github.com/google/benchmark          |                           |
| JSON for Modern C++           | MIT        | 3.6.1    | https://github.com/nlohmann/json             |                           |
| Tet-tet intersection          | Custom     |          | https://github.com/erich666/jgt-code/blob/master/Volume_07/Number_2/Ganovelli2002/tet_a_tet.h | Refactored implementation. The license is included [here](./doc/license_ganovelli_tet_a_tet.md) |

The SIMD module uses support library to provide trigonometry, logarithm and power functions. The SSE and NEON implementations are taken directly from [Julien Pommier's homepage](http://gruntthepeon.free.fr/ssemath/). The AVX version is thanks to Giovanni Garberoglio and the AVX512 to [JishinMaster](https://github.com/JishinMaster/simd_utils).
