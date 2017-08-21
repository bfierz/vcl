
# Configure the VCL libraries
INCLUDE(../cmake/VCLCudaSupport.cmake)
INCLUDE(../cmake/VCLOpenCLSupport.cmake)

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

# Library containing concrete compute classes for OpenCL
IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	SUBDIRS(vcl.compute.opencl)
	SUBDIRS(vcl.core.opencl)
	SUBDIRS(vcl.math.opencl)
ENDIF()

# Library containing concrete compute classes for CUDA
IF(CUDA_FOUND AND VCL_CUDA_SUPPORT)
	SUBDIRS(vcl.compute.cuda)
	SUBDIRS(vcl.core.cuda)
	SUBDIRS(vcl.math.cuda)
ENDIF()

