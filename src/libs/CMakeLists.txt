# The core library
SUBDIRS(vcl.core)

# Library containing mathematical routines
SUBDIRS(vcl.math)

# Library containing geometric algorithms
SUBDIRS(vcl.geometry)

# Library containing graphics classes
SUBDIRS(vcl.graphics)

# Library containing abstracted compute classes
SUBDIRS(vcl.compute)

# Library containing concrete compute classes for OpenCL
IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	SUBDIRS(vcl.compute.opencl)
ENDIF()
