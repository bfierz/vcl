#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2021 Basil Fierz
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
project(spatial_accelleration_structures)

# Status message
message(STATUS "Configuring 'Spatial Accelleration Structure Sample'")

include(../../cmake/VCLCompileGLSL.cmake)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(IMGUI
	../3rdparty/imgui/backends/imgui_impl_opengl3.cpp
	../3rdparty/imgui/backends/imgui_impl_opengl3.h
	../3rdparty/imgui/backends/imgui_impl_glfw.cpp
	../3rdparty/imgui/backends/imgui_impl_glfw.h
	../3rdparty/imgui/imconfig.h
	../3rdparty/imgui/imgui.cpp
	../3rdparty/imgui/imgui.h
	../3rdparty/imgui/imgui_demo.cpp
	../3rdparty/imgui/imgui_draw.cpp
	../3rdparty/imgui/imgui_internal.h
	../3rdparty/imgui/imgui_tables.cpp
	../3rdparty/imgui/imgui_widgets.cpp
)
set(IMFILE_DIALOG
	../3rdparty/imfiledialog/ImFileDialog.cpp
	../3rdparty/imfiledialog/ImFileDialog.h
)

set(COMMON
	../opengl/common/app.cpp
	../opengl/common/app.h
	../opengl/common/imguiapp.cpp
	../opengl/common/imguiapp.h
)

set(SOURCE_FILES
	main.cpp
	
	shaders/boundingbox.h
	shaders/boundingbox.vert
	shaders/boundingbox.frag
	shaders/trimesh.h
	shaders/trimesh.vert
	shaders/trimesh.frag
)

# Access the include directories in order to compile the HLSL shaders
get_property(CURR_INC_DIRS_0 TARGET vcl_graphics PROPERTY INCLUDE_DIRECTORIES)
list(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CURR_INC_DIRS_0} "${CMAKE_CURRENT_SOURCE_DIR}/../opengl/common/shaders")

vclcompileglsl(${PROJECT_SOURCE_DIR}/shaders/boundingbox.vert "opengl" "BoundingBoxVert" "${CURR_INC_DIRS}" COMPILEDSHADERS_0)
vclcompileglsl(${PROJECT_SOURCE_DIR}/shaders/boundingbox.frag "opengl" "BoundingBoxFrag" "${CURR_INC_DIRS}" COMPILEDSHADERS_1)
vclcompileglsl(${PROJECT_SOURCE_DIR}/shaders/trimesh.vert "opengl" "TriMeshVert" "${CURR_INC_DIRS}" COMPILEDSHADERS_2)
vclcompileglsl(${PROJECT_SOURCE_DIR}/shaders/trimesh.frag "opengl" "TriMeshFrag" "${CURR_INC_DIRS}" COMPILEDSHADERS_3)

set(COMPILEDSHADERS ${COMPILEDSHADERS_0} ${COMPILEDSHADERS_1} ${COMPILEDSHADERS_2} ${COMPILEDSHADERS_3})

add_executable(spatial "")
set_target_properties(spatial PROPERTIES FOLDER examples/opengl)
vcl_target_sources(spatial "../3rdparty" ${IMGUI})
vcl_target_sources(spatial "../3rdparty" ${IMFILE_DIALOG})
vcl_target_sources(spatial "../opengl" ${COMMON})
vcl_target_sources(spatial "" ${SOURCE_FILES})
target_sources(spatial PRIVATE ${COMPILEDSHADERS})
source_group("generated" FILES ${COMPILEDSHADERS})
target_include_directories(spatial
	PRIVATE
		../3rdparty
		../3rdparty/imgui
		../3rdparty/imgui/backends
		../3rdparty/imfiledialog
		../3rdparty/stb
)
target_link_libraries(spatial
	vcl.geometry
	vcl_graphics
	glfw
)
