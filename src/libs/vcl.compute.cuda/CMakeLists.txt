#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2015 Basil Fierz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
project(vcl_compute_cuda)

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
TARGET_INCLUDE_DIRECTORIES(vcl_compute_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_TOOLKIT_INCLUDE})

# Control OpenGL support
IF(VCL_OPENGL_SUPPORT AND TARGET ${vcl_ext_glew})
	FIND_PACKAGE(OpenGL REQUIRED)
	TARGET_INCLUDE_DIRECTORIES(vcl_compute_cuda PUBLIC ${GLEW_INCLUDE_DIR})
	TARGET_COMPILE_DEFINITIONS(vcl_compute_cuda PUBLIC VCL_OPENGL_SUPPORT)
ENDIF()

TARGET_LINK_LIBRARIES(vcl_compute_cuda
	vcl_core
	vcl_compute
	${CUDA_CUDA_LIBRARY}
)
add_dependencies(vcl_compute_cuda cuc cui)

# Setup installation
INSTALL(FILES ${VCL_CONFIG_INC} DESTINATION include/vcl/config)
INSTALL(FILES ${VCL_COMPUTE_CUDA_INC} DESTINATION include/vcl/compute/cuda)
INSTALL(TARGETS vcl_compute_cuda ARCHIVE DESTINATION lib)
