
# Configure the VCL libraries
INCLUDE(../cmake/VCLCudaSupport.cmake)
INCLUDE(../cmake/VCLOpenCLSupport.cmake)

# Tests for the core library
SUBDIRS(vcl.core)
IF(VCL_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
    SETUP_TARGET_FOR_COVERAGE(vcl_core_coverage vcl_core_test coverage_core)
ENDIF()

# Test for the library containing geometric algorithms
SUBDIRS(vcl.geometry)
IF(VCL_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
    SETUP_TARGET_FOR_COVERAGE(vcl_geometry_coverage vcl_geometry_test coverage_geometry)
ENDIF()

# Test for the library containing math routines
SUBDIRS(vcl.math)
IF(VCL_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
    SETUP_TARGET_FOR_COVERAGE(vcl_math_coverage vcl_math_test coverage_math)
ENDIF()

# Tests for the components library
SUBDIRS(vcl.components)
IF(VCL_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
    SETUP_TARGET_FOR_COVERAGE(vcl_components_coverage vcl_components_test coverage_components)
ENDIF()

# OpenGL related tests
IF(VCL_OPENGL_SUPPORT)
	FIND_PACKAGE(OpenGL REQUIRED)
	FIND_PACKAGE(GLEW REQUIRED)
	IF(GLEW_FOUND)
		SUBDIRS(vcl.graphics.opengl)
	ENDIF()
ENDIF()

# CUDA related tests
IF(CUDA_FOUND AND VCL_CUDA_SUPPORT)
	SUBDIRS(vcl.compute.cuda)
	SUBDIRS(vcl.math.cuda)
	SUBDIRS(vcl.physics.fluid)
ENDIF()

# OpenCL related tests
IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	SUBDIRS(vcl.compute.opencl)
ENDIF()
