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
project(drawtetmesh)

include(../../../cmake/VCLCompileHLSL.cmake)

# Status message
message(STATUS "Configuring 'D3D12 DrawTetMesh'")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(COMMON
	../common/app.cpp
	../common/app.h
)

set(SOURCE_FILES
	main.cpp

	tettotrimesh.cs.hlsl
	tetmesh.vs.hlsl
	tetmesh.gs.hlsl
	trimesh.vs.hlsl
	simple.ps.hlsl
	
)

# Access the include directories in order to compile the HLSL shaders
get_property(CURR_INC_DIRS_0 TARGET vcl_graphics PROPERTY INCLUDE_DIRECTORIES)
list(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CURR_INC_DIRS_0})

vclcompilehlsl(
	${PROJECT_SOURCE_DIR}/tettotrimesh.cs.hlsl
	"cs_6_0"
	"main"
	"TetToTriMeshCsoCS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_0
)
vclcompilehlsl(
	${PROJECT_SOURCE_DIR}/trimesh.vs.hlsl
	"vs_6_0"
	"main"
	"TriMeshCsoVS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_1
)
vclcompilehlsl(
	${PROJECT_SOURCE_DIR}/tetmesh.vs.hlsl
	"vs_6_0"
	"main"
	"TetMeshCsoVS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_2
)
vclcompilehlsl(
	${PROJECT_SOURCE_DIR}/tetmesh.gs.hlsl
	"gs_6_0"
	"main"
	"TetMeshCsoGS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_3
)
vclcompilehlsl(
	${PROJECT_SOURCE_DIR}/simple.ps.hlsl
	"ps_6_0"
	"main"
	"SimpleCsoPS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_9
)
set(COMPILEDSHADERS ${COMPILEDSHADERS_0} ${COMPILEDSHADERS_1} ${COMPILEDSHADERS_2} ${COMPILEDSHADERS_3} ${COMPILEDSHADERS_9})

add_executable(d3d12_drawtetmesh "")
vcl_target_sources(d3d12_drawtetmesh ".." ${COMMON})
vcl_target_sources(d3d12_drawtetmesh "" ${SOURCE_FILES})
target_sources(d3d12_drawtetmesh PRIVATE ${COMPILEDSHADERS})
source_group("generated" FILES ${COMPILEDSHADERS})
target_link_libraries(d3d12_drawtetmesh
	vcl.geometry
	vcl_graphics
)
