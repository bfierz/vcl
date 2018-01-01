CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0)

PROJECT(VisualComputingLibrary)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Configure all projects
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

################################################################################
# Add the externals
################################################################################
# Configure Eigen3 submodule
SET(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/externals/eigen")
find_package(Eigen3 3.3 REQUIRED)

SET(BUILD_TESTING OFF)
SET(BUILD_EXAMPLES OFF)
SUBDIRS(externals/cctz)
SUBDIRS(externals/abseil)

SET(FMT_TEST OFF)
SET(FMT_USE_CPP11 ON)
SUBDIRS(externals/fmtlib)

SUBDIRS(externals/gsl)

SET(BuildTests OFF)
SUBDIRS(externals/json)

################################################################################
# The libraries
################################################################################
SUBDIRS(libs)

# Benchmarks
OPTION(VCL_BUILD_BENCHMARKS "Build the benchmarks" ON)
IF (VCL_BUILD_BENCHMARKS)
	SUBDIRS(externals/gbenchmark)
	SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
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
