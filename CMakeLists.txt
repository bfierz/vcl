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
cmake_minimum_required(VERSION 3.23)

project(VisualComputingLibrary)

# Configure all projects
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
	SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
	SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
endif ()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")

# The C++ standard (17 or newer) comes from the Conan profile's compiler.cppstd, which
# CMakeToolchain emits as CMAKE_CXX_STANDARD. It is deliberately not set here.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For clang on Windows globally ignore unused commandline arguments
if (WIN32 AND ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument -Wno-error=deprecated-declarations")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

################################################################################
# Find the dependencies
################################################################################
# Core dependencies come from Conan (docs/adr/0001-conan-with-find-package-seam.md)
find_package(Eigen3 CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.11.0") 
	cmake_policy(SET CMP0072 NEW)
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

# Install rules and the exported vclConfig.cmake
if(PROJECT_IS_TOP_LEVEL)
	include(VCLInstall)
endif()

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
	find_package(benchmark CONFIG REQUIRED)
	find_package(foonathan_memory CONFIG REQUIRED)
	find_package(GeometricToolsEngine CONFIG REQUIRED)
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
	find_package(GTest CONFIG REQUIRED)
	find_package(nlohmann_json CONFIG REQUIRED)

	add_subdirectory(src/tests)
endif()
