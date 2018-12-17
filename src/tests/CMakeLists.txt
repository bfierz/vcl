#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2014 Basil Fierz
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

include(../cmake/VCLCudaSupport.cmake)
include(../cmake/VCLOpenCLSupport.cmake)

# Enable CMake test support
enable_testing()

add_subdirectory(vcl.core)
add_subdirectory(vcl.geometry)
add_subdirectory(vcl.math)
add_subdirectory(vcl.components)

# OpenGL related tests
IF(VCL_OPENGL_SUPPORT)
	find_package(OpenGL REQUIRED)
	find_package(GLEW REQUIRED)
	IF(GLEW_FOUND)
		add_subdirectory(vcl.graphics.opengl)
	ENDIF()
ENDIF()

# CUDA related tests
IF(CUDA_FOUND AND VCL_CUDA_SUPPORT)
	add_subdirectory(vcl.compute.cuda)
	add_subdirectory(vcl.math.cuda)
ENDIF()

# OpenCL related tests
IF(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	add_subdirectory(vcl.compute.opencl)
ENDIF()
