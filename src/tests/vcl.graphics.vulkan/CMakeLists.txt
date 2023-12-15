#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2018 Basil Fierz
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
project(vcl.graphics.vulkan.test)

INCLUDE(../../cmake/VCLCompileGLSL.cmake)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Generate test executable
add_executable(vcl.graphics.vulkan.test "")

set(VCL_TEST_SRC
	backbuffer.cpp
	buffers.cpp
	compute.cpp
	context.cpp
	descriptorset.cpp
	shaderprogram.cpp
)
set(VCL_TEST_INC
	test.h
)

set(SHADERS
	quad.vert
	quad.frag
	saxpy.comp
)

# Access the include directories in order to compile the GLSL shaders
get_property(CURR_INC_DIRS_0 TARGET vcl_graphics PROPERTY INCLUDE_DIRECTORIES)
list(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CURR_INC_DIRS_0})

vclcompileglsl(
	${PROJECT_SOURCE_DIR}/quad.vert
	"vulkan"
	"QuadSpirvVS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_0
)
vclcompileglsl(
	${PROJECT_SOURCE_DIR}/quad.frag
	"vulkan"
	"QuadSpirvFS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_1
)
vclcompileglsl(
	${PROJECT_SOURCE_DIR}/saxpy.comp
	"vulkan"
	"SaxpySpirvCS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_2
)
set(COMPILEDSHADERS ${COMPILEDSHADERS_0} ${COMPILEDSHADERS_1} ${COMPILEDSHADERS_2})

target_sources(vcl.graphics.vulkan.test
	PRIVATE
		${VCL_TEST_INC}
		${VCL_TEST_SRC}
		${COMPILEDSHADERS}
)

source_group("" FILES ${VCL_TEST_SRC} ${VCL_TEST_INC})
set_target_properties(vcl.graphics.vulkan.test PROPERTIES FOLDER tests)

target_link_libraries(vcl.graphics.vulkan.test
	vcl_graphics
	gtest
	gtest_main
)
