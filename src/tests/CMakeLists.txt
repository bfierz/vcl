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
project(vcl.test)

include(../cmake/VCLCudaSupport.cmake)
include(../cmake/VCLOpenCLSupport.cmake)

# Enable CMake test support
enable_testing()

add_subdirectory(vcl.core)
add_subdirectory(vcl.geometry)
add_subdirectory(vcl.math)
add_subdirectory(vcl.components)
set(test_project vcl.core.test vcl.geometry.test vcl.math.test vcl.components.test)

# D3D12 related tests
if(VCL_D3D12_SUPPORT)
	add_subdirectory(vcl.graphics.d3d12)
	list(APPEND test_project vcl.graphics.d3d12)
endif()

# OpenGL related tests
if(VCL_OPENGL_SUPPORT)
	add_subdirectory(vcl.graphics.opengl)
	list(APPEND test_project vcl.graphics.opengl)
endif()

# WebGPU related tests
if(VCL_WEBGPU_SUPPORT)
	add_subdirectory(vcl.graphics.webgpu)
	list(APPEND test_project vcl.graphics.webgpu)
endif()

# CUDA related tests
if(CUDA_FOUND AND VCL_CUDA_SUPPORT)
	add_subdirectory(vcl.compute.cuda)
	add_subdirectory(vcl.math.cuda)
	list(APPEND test_project vcl.compute.cuda vcl.math.cuda)
endif()

# OpenCL related tests
if(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	add_subdirectory(vcl.compute.opencl)
	list(APPEND test_project vcl.compute.opencl)
endif()

# Setup a target for ctest
if(VCL_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
	setup_target_for_coverage_gcovr_xml(
		NAME vcl.test
		EXECUTABLE ctest --no-compress-output -T Test
		DEPENDENCIES
			vcl.core.test
			vcl.geometry.test
			vcl.math.test
			vcl.components.test
	)
elseif(VCL_CODE_COVERAGE AND MSVC)
	find_program(
		OPENCPPCOVERAGE_PATH
			OpenCppCoverage
		HINTS
			"C:\\Program Files\\OpenCppCoverage"
	)
	
	if(NOT OPENCPPCOVERAGE_PATH)
		message(FATAL_ERROR "OpenCppCoverage not found! Aborting...")
	endif()
	
	set(OPENCPPCOVERAGE_SOURCES "${CMAKE_SOURCE_DIR}")
	string(REGEX REPLACE "/" "\\\\" OPENCPPCOVERAGE_SOURCES ${OPENCPPCOVERAGE_SOURCES})
	add_custom_target(vcl.test
		COMMAND ${OPENCPPCOVERAGE_PATH} --sources "${OPENCPPCOVERAGE_SOURCES}" --modules vcl* --export_type=cobertura --cover_children -- ctest --no-compress-output -T Test

		WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
		DEPENDS
			vcl.core.test
			vcl.geometry.test
			vcl.math.test
			vcl.components.test
		COMMENT "Generate code coverage report using OpenCppCoverage"
	)
else()
	add_custom_target(vcl.test
				COMMAND ctest --no-compress-output -T Test

		WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
		DEPENDS
			vcl.core.test
			vcl.geometry.test
			vcl.math.test
			vcl.components.test
	)
endif()
set_target_properties(vcl.test PROPERTIES FOLDER tests)
