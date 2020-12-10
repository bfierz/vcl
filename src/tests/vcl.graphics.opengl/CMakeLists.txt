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
project(vcl.graphics.opengl.test)

INCLUDE(../../cmake/VCLCompileGLSL.cmake)
include(../../cmake/VCLTest.cmake)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(SOURCE_FILES
	bindings.cpp
	bitmap.h
	blendstate.cpp
	buffers.cpp
	depthstencilstate.cpp
	engine.cpp
	histogram.cpp
	imageprocessor.cpp
	inputlayout.cpp
	main.cpp
	pattern.h
	radixsort.cpp
	rasterstate.cpp
	scan.cpp
	shaderprogram.cpp
	textures.cpp
	
	quad.vert
	quad.frag
)

# Access the include directories in order to compile the GLSL shaders
get_property(CURR_INC_DIRS_0 TARGET vcl_graphics PROPERTY INCLUDE_DIRECTORIES)
list(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CURR_INC_DIRS_0})

vclcompileglsl(
	${PROJECT_SOURCE_DIR}/quad.vert
	"opengl"
	"QuadSpirvVS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_0
)
vclcompileglsl(
	${PROJECT_SOURCE_DIR}/quad.frag
	"opengl"
	"QuadSpirvFS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_1
)
set(COMPILEDSHADERS ${COMPILEDSHADERS_0} ${COMPILEDSHADERS_1})

vcl_add_test(vcl.graphics.opengl.test)
vcl_target_sources(vcl.graphics.opengl.test "" ${SOURCE_FILES})
target_sources(vcl.graphics.opengl.test PRIVATE ${COMPILEDSHADERS})
target_link_libraries(vcl.graphics.opengl.test
	vcl_graphics
)
