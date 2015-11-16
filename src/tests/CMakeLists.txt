
# Set the paths to the google test libraries
SET(VCL_GTEST_INCLUDE_DIR CACHE PATH "Include directory of gtest")
SET(VCL_GTEST_LIBRARY_DEBUG CACHE PATH "Library directory containing the gtest debug libraries")
SET(VCL_GTEST_LIBRARY_RELEASE CACHE PATH "Library directory containing the gtest release libraries")
INCLUDE_DIRECTORIES(${VCL_GTEST_INCLUDE_DIR})

# Tests for the core library
SUBDIRS(vcl.core)

# Test for the library containing geometric algorithms
SUBDIRS(vcl.geometry)

# Test for the library containing math routines
SUBDIRS(vcl.math)

# Tests for the components library
SUBDIRS(vcl.components)

# OpenGL related tests
IF(GLEW_FOUND AND VCL_OPENGL_SUPPORT)
	SUBDIRS(vcl.graphics.opengl)
ENDIF()

# OpenCL related tests
IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	SUBDIRS(vcl.compute.opencl)
ENDIF()
