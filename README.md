VCL
===

[![Build Status](https://travis-ci.org/bfierz/vcl.svg?branch=master)](https://travis-ci.org/bfierz/vcl)
[![Build status](https://ci.appveyor.com/api/projects/status/ul6ci6u6t2wgyes7?svg=true)](https://ci.appveyor.com/project/bfierz/vcl)
[![Build Status](https://dev.azure.com/basilfierz/VCL/_apis/build/status/vcl?branchName=master)](https://dev.azure.com/basilfierz/VCL/_build/latest?definitionId=2&branchName=master)
[![codecov](https://codecov.io/gh/bfierz/vcl/branch/master/graph/badge.svg)](https://codecov.io/gh/bfierz/vcl)

The Visual Computing Library (VCL) started as a repository for algorithms and data structures developed during my PhD student time. As the name says it was used to visual computing techniques, which in this case were visual simulations.

Requirements
------------

Assumes C++14 capable compiler with some compatibility for older MSVC versions.

C++ feature matrices:
A good overview on the feature support for different compiler versions can be found here: https://en.cppreference.com/w/cpp/compiler_support

Detailed reports at the vendor's pages:

* MSVC: https://msdn.microsoft.com/en-us/library/hh567368.aspx, https://docs.microsoft.com/en-us/cpp/visual-cpp-language-conformance?view=vs-2017
* Intel: https://software.intel.com/en-us/articles/c14-features-supported-by-intel-c-compiler
* Clang: https://clang.llvm.org/cxx_status.html
* GCC: https://gcc.gnu.org/projects/cxx-status.html

Supported compilers:

* MSVC >= 2015
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
| Abseil Common Libraries (C++) | Apache-2.0 | 284378a  | https://abseil.io                            | Release 20181200 + bugfix |
| C++ commandline parsing       | MIT        | 2.1.2    | https://github.com/jarro2783/cxxopts         | In source tree            |
| Eigen 3                       | MPL2       | 3.3.7    | https://eigen.tuxfamily.org                  |                           |
| Expected lite                 | BSL-1.0    | 0.2.0    | https://github.com/martinmoene/expected-lite | In source tree            |
| Google Test                   | BSD-3      | 1.8.1    | https://github.com/google/googletest         |                           |
| Google Benchmark              | Apache-2.0 | 1.4.1    | https://github.com/google/benchmark          |                           |
| JSON for Modern C++           | MIT        | 3.6.1    | https://github.com/nlohmann/json             |                           |
