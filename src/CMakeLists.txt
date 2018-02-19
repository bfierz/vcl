#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2014 Basil Fierz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
cmake_minimum_required(VERSION 3.5.0)

project(VisualComputingLibrary)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Conan setup
option(VCL_USE_CONAN OFF)
if (VCL_USE_CONAN)
	include(${CMAKE_SOURCE_DIR}/cmake/VCLConanIntegration.cmake)
	include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
	conan_basic_setup(TARGETS)
endif()


# Configure all projects
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Define C++ standard, minimum requirement is C++14
# As MSVC is not able to define the minimum level, software needs
# to implement per feature detection
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For clang on Windows globally ignore unused commandline arguments
if (WIN32 AND ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument -Wno-error=deprecated-declarations")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

################################################################################
# Add the externals
################################################################################
# Configure Eigen3 submodule
if (VCL_USE_CONAN)
	vcl_copy_conan_target(Eigen3 CONAN_PKG::eigen)
else()
	set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/externals/eigen")
	find_package(Eigen3 3.3 QUIET)
endif()

if (VCL_USE_CONAN)
	set(absl "CONAN_PKG::abseil")
else()
	option(BUILD_TESTING OFF)
	option(BUILD_EXAMPLES OFF)
	subdirs(externals/cctz)
	subdirs(externals/abseil)

	set(absl
		absl::any
		absl::bad_any_cast
	)
endif()

if (VCL_USE_CONAN)
	vcl_copy_conan_target(fmt-header-only CONAN_PKG::fmt)
else()
	option(FMT_TEST OFF)
	option(FMT_USE_CPP11 ON)
	subdirs(externals/fmtlib)
endif()

if (VCL_USE_CONAN)
	vcl_copy_conan_target(GSL CONAN_PKG::gsl_microsoft)
else()
	subdirs(externals/gsl)
endif()

OPTION(FOONATHAN_MEMORY_BUILD_EXAMPLES OFF)
OPTION(FOONATHAN_MEMORY_BUILD_TESTS OFF)
SUBDIRS(externals/memory)

OPTION(BuildTests OFF)
SUBDIRS(externals/json)

FIND_PACKAGE(EGL QUIET)
IF(EGL_FOUND)
	ADD_LIBRARY(egl UNKNOWN IMPORTED)
	SET_PROPERTY(TARGET egl PROPERTY TARGET_INCLUDE_DIRECTORIES "${EGL_INCLUDE_DIR}")
	SET_PROPERTY(TARGET egl PROPERTY IMPORTED_LOCATION "${EGL_LIBRARY}")
ELSE()
	SUBDIRS(externals/EGL/EGL)
ENDIF()

################################################################################
# The libraries
################################################################################
SUBDIRS(libs)

# Benchmarks
OPTION(VCL_BUILD_BENCHMARKS "Build the benchmarks" ON)
IF (VCL_BUILD_BENCHMARKS)
	OPTION(BENCHMARK_ENABLE_TESTING OFF)
	SUBDIRS(externals/gbenchmark)
	#SET_TARGET_PROPERTIES(benchmark PROPERTIES FOLDER 3rd-party)
	SUBDIRS(benchmarks)
ENDIF (VCL_BUILD_BENCHMARKS)

# Examples
OPTION(VCL_BUILD_EXAMPLES "Build the examples" ON)
IF (VCL_BUILD_EXAMPLES)
	SUBDIRS(examples)
ENDIF (VCL_BUILD_EXAMPLES)

# Tools
OPTION(VCL_BUILD_TOOLS "Build the tools" ON)
IF (VCL_BUILD_TOOLS)
	SUBDIRS(tools)
ENDIF (VCL_BUILD_TOOLS)

# Unit tests
OPTION(VCL_BUILD_TESTS "Build the unit tests" OFF)
IF (VCL_BUILD_TESTS)
	SET(BUILD_GTEST ON CACHE BOOL "" FORCE)
	SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	SUBDIRS(externals/googletest)

	SET_PROPERTY(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS GTEST_LANG_CXX11=1)
	SUBDIRS(tests)
	
ENDIF (VCL_BUILD_TESTS)
