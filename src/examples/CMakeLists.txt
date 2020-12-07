
# Configure the VCL libraries
INCLUDE(../cmake/VCLCudaSupport.cmake)
INCLUDE(../cmake/VCLOpenCLSupport.cmake)

SUBDIRS(rtti)
SUBDIRS(simd)

if(VCL_OPENGL_SUPPORT)
	add_subdirectory(3rdparty/glfw EXCLUDE_FROM_ALL)
	set_property(TARGET glfw PROPERTY FOLDER 3rd-party)
endif()

IF(VCL_OPENGL_SUPPORT)
	SUBDIRS(meshviewer)
	SUBDIRS(opengl)
ENDIF()

IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	SUBDIRS(opencl)
ENDIF()
