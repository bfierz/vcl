CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

PROJECT(VisualComputingLibrary)

INCLUDE(cmake/VCLConfig.cmake)

# Configure all projects
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# The libraries
SUBDIRS(libs)

# Examples
SUBDIRS(examples)

# Tools
SUBDIRS(tools)

# Unit tests
OPTION(VCL_BUILD_TESTS "Build the unit tests" OFF)

IF (VCL_BUILD_TESTS)
	SUBDIRS(tests)
ENDIF (VCL_BUILD_TESTS)
