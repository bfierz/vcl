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
cmake_minimum_required(VERSION 3.11.0)

project(VisualComputingLibrary)

# Determine whether this is a standalone project or included by other projects
set(VCL_STANDALONE_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
	set(VCL_STANDALONE_PROJECT ON)

	# Configure all projects
	SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
	SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
endif ()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")

# Conan setup
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/VCLTargets.cmake)

# Define C++ standard, minimum requirement is C++14
# As MSVC is not able to define the minimum level, software needs
# to implement per feature detection
set(VCL_CXX_STANDARD "14" CACHE STRING "C++ standard")
set_property(CACHE VCL_CXX_STANDARD PROPERTY STRINGS "14" "17" "20")

set(CMAKE_CXX_STANDARD ${VCL_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For clang on Windows globally ignore unused commandline arguments
if (WIN32 AND ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument -Wno-error=deprecated-declarations")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Global configurations for emscripten
if (EMSCRIPTEN)
	# Make sure exception can be used
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s DISABLE_EXCEPTION_CATCHING=0")
endif(EMSCRIPTEN)

################################################################################
# Add the externals
################################################################################
if (NOT vcl_ext_eigen)
	set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/externals/eigen")
	find_package(Eigen3 3.3 QUIET)
	vcl_copy_target(vcl_ext_eigen_tgt Eigen3)
	set(vcl_ext_eigen "vcl_ext_eigen_tgt")
endif()

if (NOT vcl_ext_absl)
	option(ABSL_BUILD_TESTING "" OFF)
	option(ABSL_PROPAGATE_CXX_STD "" ON)
	add_subdirectory(src/externals/abseil EXCLUDE_FROM_ALL)
	vcl_combine_targets(vcl_ext_absl_tgt absl::any absl::bad_any_cast)
	set(vcl_ext_absl "vcl_ext_absl_tgt")
	
	set_property(TARGET absl_bad_any_cast_impl PROPERTY FOLDER 3rd-party/absl)
	set_property(TARGET absl_log_severity PROPERTY FOLDER 3rd-party/absl)
	set_property(TARGET absl_raw_logging_internal PROPERTY FOLDER 3rd-party/absl)
endif()

if (NOT vcl_ext_fmt)
	option(FMT_TEST "" OFF)
	option(FMT_USE_CPP11 "" ON)
	add_subdirectory(src/externals/fmtlib EXCLUDE_FROM_ALL)
	vcl_copy_target(vcl_ext_fmt_tgt fmt::fmt-header-only)
	set(vcl_ext_fmt "vcl_ext_fmt_tgt")
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/externals/json/CMakeLists.txt")
	option(JSON_BuildTests "" OFF)
	option(JSON_Install "" OFF)
	add_subdirectory(src/externals/json)
endif()

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.11.0") 
	cmake_policy(SET CMP0072 NEW)
endif()
find_package(OpenGL QUIET)
if(OpenGL_FOUND AND NOT EMSCRIPTEN)
	find_package(EGL QUIET)
	if(EGL_FOUND)
		add_library(vcl_ext_egl_tgt UNKNOWN IMPORTED)
		set_property(TARGET vcl_ext_egl_tgt PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${EGL_INCLUDE_DIR}")
		set_property(TARGET vcl_ext_egl_tgt PROPERTY IMPORTED_LOCATION "${EGL_LIBRARY}")
		set(vcl_ext_egl "vcl_ext_egl_tgt")
	endif()
endif()

################################################################################
# The libraries
################################################################################
# Control code coverage measurement
set(VCL_CODE_COVERAGE CACHE BOOL "Enable code coverage measurement")

if(VCL_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
	include(CodeCoverage)
	append_coverage_compiler_flags()
endif()

add_subdirectory(src/libs)

# Build options
if("^${CMAKE_SOURCE_DIR}$" STREQUAL "^${PROJECT_SOURCE_DIR}$")
	option(VCL_BUILD_BENCHMARKS "Build the benchmarks" ON)
	option(VCL_BUILD_EXAMPLES "Build the examples" OFF)
	option(VCL_BUILD_TESTS "Build the unit tests" ON)
	option(VCL_BUILD_TOOLS "Build the tools" ON)
else()
	option(VCL_BUILD_BENCHMARKS "Build the benchmarks" OFF)
	option(VCL_BUILD_EXAMPLES "Build the examples" OFF)
	option(VCL_BUILD_TESTS "Build the unit tests" OFF)
	option(VCL_BUILD_TOOLS "Build the tools" ON)
endif()

# Benchmarks
if(VCL_BUILD_BENCHMARKS)
	option(BENCHMARK_ENABLE_TESTING "" OFF)
	option(BENCHMARK_ENABLE_GTEST_TESTS "" OFF)
	add_subdirectory(src/externals/gbenchmark)
	set_property(TARGET benchmark PROPERTY FOLDER 3rd-party)
	set_property(TARGET benchmark_main PROPERTY FOLDER 3rd-party)
	if (WIN32 AND ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
		set_property(TARGET benchmark PROPERTY COMPILE_OPTIONS "/W4")
	endif()
	add_subdirectory(src/benchmarks)
endif()

# Examples
if(VCL_BUILD_EXAMPLES)
	add_subdirectory(src/examples)
endif()

# Tools
if(VCL_BUILD_TOOLS)
	add_subdirectory(src/tools)
endif()

# Unit tests
if(VCL_BUILD_TESTS)
	option(gtest_build_samples "" OFF)
	option(gtest_build_tests "" OFF)
	option(gtest_force_shared_crt "" ON)
	add_subdirectory(src/externals/googletest EXCLUDE_FROM_ALL)
	set_property(TARGET gtest PROPERTY FOLDER 3rd-party)
	set_property(TARGET gtest_main PROPERTY FOLDER 3rd-party)

	# Remove non-supported options
	if (WIN32 AND ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
		vcl_replace_compile_flag(gtest "[-/]W." "/W0")
		vcl_replace_compile_flag(gtest_main "[-/]W." "/W0")
		vcl_remove_compile_flag(gtest "/Gm-")
		vcl_remove_compile_flag(gtest_main "/Gm-")
	endif()
	
	set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS GTEST_LANG_CXX11=1)
	add_subdirectory(src/tests)
endif()
