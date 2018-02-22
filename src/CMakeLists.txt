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
include(${CMAKE_SOURCE_DIR}/cmake/VCLConanIntegration.cmake)

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
if (NOT vcl_ext_eigen)
	set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/externals/eigen")
	find_package(Eigen3 3.3 QUIET)
	vcl_copy_target(vcl_ext_eigen_tgt Eigen3)
	set(vcl_ext_eigen "vcl_ext_eigen_tgt")
endif()

if (NOT vcl_ext_absl)
	option(BUILD_TESTING OFF)
	option(BUILD_EXAMPLES OFF)
	add_subdirectory(externals/cctz)
	add_subdirectory(externals/abseil)
	vcl_combine_targets(vcl_ext_absl_tgt absl::any absl::bad_any_cast)
	set(vcl_ext_absl "vcl_ext_absl_tgt")
endif()

if (NOT vcl_ext_fmt)
	option(FMT_TEST OFF)
	option(FMT_USE_CPP11 ON)
	add_subdirectory(externals/fmtlib)	
	vcl_copy_target(vcl_ext_fmt_tgt fmt::fmt-header-only)
	set(vcl_ext_fmt "vcl_ext_fmt_tgt")
endif()

if (NOT vcl_ext_gsl)
	add_subdirectory(externals/gsl)
	vcl_copy_target(vcl_ext_gsl_tgt GSL)
	set(vcl_ext_gsl "vcl_ext_gsl_tgt")
endif()

OPTION(FOONATHAN_MEMORY_BUILD_EXAMPLES OFF)
OPTION(FOONATHAN_MEMORY_BUILD_TESTS OFF)
add_subdirectory(externals/memory)

OPTION(BuildTests OFF)
add_subdirectory(externals/json)

FIND_PACKAGE(EGL QUIET)
IF(EGL_FOUND)
	ADD_LIBRARY(egl UNKNOWN IMPORTED)
	SET_PROPERTY(TARGET egl PROPERTY TARGET_INCLUDE_DIRECTORIES "${EGL_INCLUDE_DIR}")
	SET_PROPERTY(TARGET egl PROPERTY IMPORTED_LOCATION "${EGL_LIBRARY}")
ELSE()
	add_subdirectory(externals/EGL/EGL)
ENDIF()

if (NOT vcl_ext_glew)
	find_package(GLEW QUIET)
	if(GLEW_FOUND)
		add_library(vcl_ext_glew_tgt UNKNOWN IMPORTED)
		set_property(TARGET vcl_ext_glew_tgt PROPERTY TARGET_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIR}")
		set_property(TARGET vcl_ext_glew_tgt PROPERTY IMPORTED_LOCATION "${GLEW_LIBRARY}")
	endif()
	set(vcl_ext_glew "vcl_ext_glew_tgt")
endif()

################################################################################
# The libraries
################################################################################
add_subdirectory(libs)

# Benchmarks
OPTION(VCL_BUILD_BENCHMARKS "Build the benchmarks" ON)
IF (VCL_BUILD_BENCHMARKS)
	OPTION(BENCHMARK_ENABLE_TESTING OFF)
	add_subdirectory(externals/gbenchmark)
	#SET_TARGET_PROPERTIES(benchmark PROPERTIES FOLDER 3rd-party)
	add_subdirectory(benchmarks)
ENDIF (VCL_BUILD_BENCHMARKS)

# Examples
OPTION(VCL_BUILD_EXAMPLES "Build the examples" ON)
IF (VCL_BUILD_EXAMPLES)
	add_subdirectory(examples)
ENDIF (VCL_BUILD_EXAMPLES)

# Tools
OPTION(VCL_BUILD_TOOLS "Build the tools" ON)
IF (VCL_BUILD_TOOLS)
	add_subdirectory(tools)
ENDIF (VCL_BUILD_TOOLS)

# Unit tests
OPTION(VCL_BUILD_TESTS "Build the unit tests" OFF)
IF (VCL_BUILD_TESTS)
	SET(BUILD_GTEST ON CACHE BOOL "" FORCE)
	SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	add_subdirectory(externals/googletest)

	SET_PROPERTY(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS GTEST_LANG_CXX11=1)
	add_subdirectory(tests)
	
ENDIF (VCL_BUILD_TESTS)
