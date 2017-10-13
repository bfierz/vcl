
# Configure the VCL libraries
INCLUDE(../cmake/VCLCudaSupport.cmake)
INCLUDE(../cmake/VCLOpenCLSupport.cmake)

SUBDIRS(rtti)
SUBDIRS(simd)

IF(VCL_OPENGL_SUPPORT)
	SUBDIRS(meshviewer)
	SUBDIRS(opengl)
ENDIF()

IF(CUDA_FOUND AND VCL_CUDA_SUPPORT)
	SUBDIRS(cuda)
ENDIF()

IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	SUBDIRS(opencl)
ENDIF()
