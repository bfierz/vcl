VCL
===

[![Build Status](https://travis-ci.org/bfierz/vcl.svg?branch=master)](https://travis-ci.org/bfierz/vcl)
[![Build status](https://ci.appveyor.com/api/projects/status/ul6ci6u6t2wgyes7?svg=true)](https://ci.appveyor.com/project/bfierz/vcl)
[![codecov](https://codecov.io/gh/bfierz/vcl/branch/master/graph/badge.svg)](https://codecov.io/gh/bfierz/vcl)

The Visual Computing Library (VCL) started as a repository for algorithms and data structures developed during my PhD student time. As the name says it was used to visual computing techniques, which in this case were visual simulations.

Requirements
------------

Assumes C++14 cabaple compiler with some compatibility for older MSVC versions.

C++ feature matrices:
MSVC: https://msdn.microsoft.com/en-us/library/hh567368.aspx
Intel: https://software.intel.com/en-us/articles/c14-features-supported-by-intel-c-compiler
Clang: https://clang.llvm.org/cxx_status.html
GCC: https://gcc.gnu.org/projects/cxx-status.html

Supported compilers:
* MSVC >= 2015
* Clang >= 3.5
* GCC >= 5
* Intel ICC >= 17

| Compiler / Library | Eigen | Google Test  | Google Benchmark  |   |
|--------------------|-------|--------------|-------------------|---|
| Visual Studio 2017 | 3.3.3 | 1.8.0        | 1.1.0             |   |

Shipped External Libraries
--------------------------

C++ commandline parsing (MIT): https://github.com/jarro2783/cxxopts
