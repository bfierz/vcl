# The core library
SUBDIRS(vcl.core)

# Library containing mathematical routines
SUBDIRS(vcl.math)

# Library containing geometric algorithms
SUBDIRS(vcl.geometry)

# Library containing graphics classes
SUBDIRS(vcl.graphics)

# Library implementing a simple entity component system
SUBDIRS(vcl.components)

# Library containing abstracted compute classes
SUBDIRS(vcl.compute)

# Control OpenCL support
SET(VCL_OPENCL_SUPPORT CACHE BOOL "Enable OpenCL support")
IF(VCL_OPENCL_SUPPORT)
	FIND_PACKAGE(OpenCL)
	IF(OPENCL_FOUND)
		ADD_DEFINITIONS(-DVCL_OPENCL_SUPPORT)
	ENDIF()
ENDIF()

# Library containing concrete compute classes for OpenCL
IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	SUBDIRS(vcl.compute.opencl)
	SUBDIRS(vcl.core.opencl)
	SUBDIRS(vcl.math.opencl)
ENDIF()

# Control CUDA support
SET(VCL_CUDA_SUPPORT CACHE BOOL "Turn off the VCL CUDA support")
IF(VCL_CUDA_SUPPORT)
	FIND_PACKAGE(CUDA)	
	IF(CUDA_FOUND)
		ADD_DEFINITIONS(-DVCL_CUDA_SUPPORT)
	ENDIF(CUDA_FOUND)
ENDIF(VCL_CUDA_SUPPORT)

# Library containing concrete compute classes for CUDA
IF(CUDA_FOUND AND VCL_CUDA_SUPPORT)
	SUBDIRS(vcl.compute.cuda)
	SUBDIRS(vcl.core.cuda)
	SUBDIRS(vcl.math.cuda)
ENDIF()

