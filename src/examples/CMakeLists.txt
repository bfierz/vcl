# Add GLFW to build graphics examples with simple windows
SUBDIRS(glfw)

# Configure the VCL libraries
INCLUDE(../cmake/VCLCudaSupport.cmake)
INCLUDE(../cmake/VCLOpenCLSupport.cmake)

SUBDIRS(rtti)
SUBDIRS(simd)

IF(VCL_OPENGL_SUPPORT)
	SUBDIRS(meshviewer)
	SUBDIRS(opengl)
ENDIF()

IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	SUBDIRS(opencl)
ENDIF()

IF(VCL_VULKAN_SUPPORT)
	SUBDIRS(vulkan)
ENDIF()
