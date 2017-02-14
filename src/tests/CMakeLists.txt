
# Tests for the core library
SUBDIRS(vcl.core)

# Test for the library containing geometric algorithms
SUBDIRS(vcl.geometry)

# Test for the library containing math routines
SUBDIRS(vcl.math)

# Tests for the components library
SUBDIRS(vcl.components)

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
ENDIF()

# OpenCL related tests
IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	SUBDIRS(vcl.compute.opencl)
ENDIF()
