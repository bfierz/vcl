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

# Status message
message(STATUS "Configuring 'OpenGL ImGui'")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(IMGUI
	../../3rdparty/imgui/backends/imgui_impl_opengl3.cpp
	../../3rdparty/imgui/backends/imgui_impl_opengl3.h
	../../3rdparty/imgui/backends/imgui_impl_glfw.cpp
	../../3rdparty/imgui/backends/imgui_impl_glfw.h
	../../3rdparty/imgui/imconfig.h
	../../3rdparty/imgui/imgui.cpp
	../../3rdparty/imgui/imgui.h
	../../3rdparty/imgui/imgui_demo.cpp
	../../3rdparty/imgui/imgui_draw.cpp
	../../3rdparty/imgui/imgui_internal.h
	../../3rdparty/imgui/imgui_widgets.cpp
)

set(COMMON
	../common/app.cpp
	../common/app.h
	../common/imguiapp.cpp
	../common/imguiapp.h
)

set(SOURCE_FILES
	main.cpp
)

add_executable(opengl_imgui "")
set_target_properties(opengl_imgui PROPERTIES FOLDER examples/opengl)
vcl_target_sources(opengl_imgui "../../3rdparty" ${IMGUI})
vcl_target_sources(opengl_imgui ".." ${COMMON})
vcl_target_sources(opengl_imgui "" ${SOURCE_FILES})
target_include_directories(opengl_imgui
	PRIVATE
		../../3rdparty/imgui
		../../3rdparty/imgui/backends
)
target_link_libraries(opengl_imgui
	vcl_graphics
	glfw
)
