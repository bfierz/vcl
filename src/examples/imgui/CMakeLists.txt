#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2026 Basil Fierz
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

# The Dear ImGui backends of the examples, compiled once for all of them. imgui ships its
# platform and renderer backends as sources, so the backends enabled in this build are the only
# things compiled here. The core library, ImPlot, ImGuizmo and ImFileDialog come as ready-built
# packages.
message(STATUS "Configuring 'vcl_examples_imgui'")

find_package(imgui CONFIG REQUIRED)

# Where the backend sources come from is up to whoever provides imgui: conanfile.py sets the
# folder of the Conan package, a system or parent-project imgui passes its 'backends' folder.
set(VCL_IMGUI_BACKEND_DIR "" CACHE PATH "Folder with the Dear ImGui backend sources (imgui_impl_*.cpp)")
if(NOT EXISTS "${VCL_IMGUI_BACKEND_DIR}/imgui_impl_glfw.cpp")
	message(FATAL_ERROR "VCL_IMGUI_BACKEND_DIR ('${VCL_IMGUI_BACKEND_DIR}') does not contain the Dear ImGui backend sources")
endif()

set(VCL_IMGUI_BACKENDS)
set(VCL_IMGUI_LIBRARIES)

# GLFW is the window system of all examples besides the Direct3D 12 ones (found in the parent folder)
if(VCL_OPENGL_SUPPORT OR VCL_VULKAN_SUPPORT OR VCL_WEBGPU_SUPPORT)
	list(APPEND VCL_IMGUI_BACKENDS imgui_impl_glfw)
	if(NOT EMSCRIPTEN)
		list(APPEND VCL_IMGUI_LIBRARIES glfw)
	endif()
endif()
if(VCL_OPENGL_SUPPORT)
	list(APPEND VCL_IMGUI_BACKENDS imgui_impl_opengl3)
endif()
if(VCL_VULKAN_SUPPORT)
	list(APPEND VCL_IMGUI_BACKENDS imgui_impl_vulkan)
endif()
if(VCL_WEBGPU_SUPPORT)
	list(APPEND VCL_IMGUI_BACKENDS imgui_impl_wgpu)
endif()
if(VCL_D3D12_SUPPORT AND WIN32)
	list(APPEND VCL_IMGUI_BACKENDS imgui_impl_win32 imgui_impl_dx12)
endif()

# Extensions of ImGui that the examples demonstrate
if(VCL_OPENGL_SUPPORT)
	find_package(implot CONFIG REQUIRED)
	list(APPEND VCL_IMGUI_LIBRARIES implot::implot)
endif()
if(VCL_OPENGL_SUPPORT OR VCL_WEBGPU_SUPPORT)
	# Pulls in stb
	find_package(imfiledialog CONFIG REQUIRED)
	list(APPEND VCL_IMGUI_LIBRARIES imfiledialog::imfiledialog)
endif()
if(VCL_D3D12_SUPPORT AND WIN32)
	find_package(imguizmo CONFIG REQUIRED)
	list(APPEND VCL_IMGUI_LIBRARIES imguizmo::imguizmo)
endif()

set(VCL_IMGUI_SOURCES)
foreach(VCL_IMGUI_BACKEND ${VCL_IMGUI_BACKENDS})
	list(APPEND VCL_IMGUI_SOURCES
		"${VCL_IMGUI_BACKEND_DIR}/${VCL_IMGUI_BACKEND}.cpp"
		"${VCL_IMGUI_BACKEND_DIR}/${VCL_IMGUI_BACKEND}.h"
	)
endforeach()

add_library(vcl_examples_imgui STATIC ${VCL_IMGUI_SOURCES})
# The backends are Dear ImGui's code, so clang-tidy leaves them out
set_target_properties(vcl_examples_imgui PROPERTIES FOLDER examples VCL_SKIP_CLANG_TIDY ON)
target_include_directories(vcl_examples_imgui PUBLIC "${VCL_IMGUI_BACKEND_DIR}")
if(VCL_WEBGPU_SUPPORT AND NOT EMSCRIPTEN)
	# Use the WebGPU API of the Dawn package
	target_compile_definitions(vcl_examples_imgui PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_DAWN)
endif()
target_link_libraries(vcl_examples_imgui
	PUBLIC
		imgui::imgui
		vcl_graphics
		${VCL_IMGUI_LIBRARIES}
)
if(MINGW AND VCL_D3D12_SUPPORT)
	# MinGW ignores the '#pragma comment(lib)' of the ImGui backends, and the
	# DX12 backend needs the interface IDs declared for __uuidof
	target_link_libraries(vcl_examples_imgui PUBLIC dwmapi d3dcompiler)
	set_source_files_properties("${VCL_IMGUI_BACKEND_DIR}/imgui_impl_dx12.cpp"
		PROPERTIES COMPILE_OPTIONS "-include;vcl/config/direct3d12.h"
	)
endif()
