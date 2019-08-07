PROJECT(vcl_compute_cuda)

# VCL / CONFIG
SET(VCL_CONFIG_INC
	vcl/config/cuda.h
)

# VCL / COMPUTE / CUDA
SET(VCL_COMPUTE_CUDA_INC
	vcl/compute/cuda/buffer.h
	vcl/compute/cuda/commandqueue.h
	vcl/compute/cuda/context.h
	vcl/compute/cuda/device.h
	vcl/compute/cuda/module.h
	vcl/compute/cuda/kernel.h
	vcl/compute/cuda/platform.h
)
SET(VCL_COMPUTE_CUDA_SRC
	vcl/compute/cuda/buffer.cpp
	vcl/compute/cuda/commandqueue.cpp
	vcl/compute/cuda/context.cpp
	vcl/compute/cuda/device.cpp
	vcl/compute/cuda/module.cpp
	vcl/compute/cuda/kernel.cpp
	vcl/compute/cuda/platform.cpp
)

SOURCE_GROUP("config" FILES ${VCL_CONFIG_INC})
SOURCE_GROUP("" FILES ${VCL_COMPUTE_CUDA_SRC} ${VCL_COMPUTE_CUDA_INC})

SET(SOURCE
	${VCL_CONFIG_INC}
	${VCL_COMPUTE_CUDA_SRC} ${VCL_COMPUTE_CUDA_INC}
)

# Generate library
ADD_LIBRARY(vcl_compute_cuda STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_compute_cuda PROPERTIES FOLDER libs)
if(WIN32)
	SET_TARGET_PROPERTIES(vcl_compute_cuda PROPERTIES DEBUG_POSTFIX _d)
endif()
TARGET_INCLUDE_DIRECTORIES(vcl_compute_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_TOOLKIT_INCLUDE} ${GLEW_INCLUDE_DIR})

# Control OpenGL support
SET(VCL_OPENGL_SUPPORT CACHE BOOL "Enable OpenGL support")

IF(VCL_OPENGL_SUPPORT)
	FIND_PACKAGE(OpenGL REQUIRED)
	FIND_PACKAGE(GLEW REQUIRED)
	IF(GLEW_FOUND)
		TARGET_COMPILE_DEFINITIONS(vcl_compute_cuda PUBLIC VCL_OPENGL_SUPPORT)
	ENDIF()
ENDIF()

TARGET_LINK_LIBRARIES(vcl_compute_cuda
	vcl_core
	vcl_compute
	${CUDA_CUDA_LIBRARY}
)
add_dependencies(vcl_compute_cuda cuc)

# Setup installation
INSTALL(FILES ${VCL_CONFIG_INC} DESTINATION include/vcl/config)
INSTALL(FILES ${VCL_COMPUTE_CUDA_INC} DESTINATION include/vcl/compute/cuda)
INSTALL(TARGETS vcl_compute_cuda ARCHIVE DESTINATION lib)
