#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2020 Basil Fierz
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
project(vcl.graphics.d3d12.test)

include(../../cmake/VCLCompileHLSL.cmake)
include(../../cmake/VCLTest.cmake)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(SOURCE_FILES
	buffers.cpp
	compute.cpp
	engine.cpp
	main.cpp
	rendering.cpp
	swapchain.cpp
	textures.cpp

	quad.vs.hlsl
	quad.ps.hlsl
	saxpy.cs.hlsl
)

# Access the include directories in order to compile the HLSL shaders
get_property(CURR_INC_DIRS_0 TARGET vcl_graphics PROPERTY INCLUDE_DIRECTORIES)
list(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CURR_INC_DIRS_0})

vclcompilehlsl(
	${PROJECT_SOURCE_DIR}/quad.vs.hlsl
	"vs_6_0"
	"main"
	"QuadCsoVS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_0
)
vclcompilehlsl(
	${PROJECT_SOURCE_DIR}/quad.ps.hlsl
	"ps_6_0"
	"main"
	"QuadCsoPS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_1
)
vclcompilehlsl(
	${PROJECT_SOURCE_DIR}/saxpy.cs.hlsl
	"cs_6_0"
	"main"
	"SaxpyCsoCS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_2
)
set(COMPILEDSHADERS ${COMPILEDSHADERS_0} ${COMPILEDSHADERS_1} ${COMPILEDSHADERS_2})

vcl_add_test(vcl.graphics.d3d12.test)
vcl_target_sources(vcl.graphics.d3d12.test "" ${SOURCE_FILES})
target_sources(vcl.graphics.d3d12.test PRIVATE ${COMPILEDSHADERS})
target_link_libraries(vcl.graphics.d3d12.test
	vcl_graphics
)
add_dependencies(vcl.graphics.d3d12.test bin2c)
