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
project(vcl_graphics)

include(FetchContent)

# Configure the VCL compilation environment
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/VCLConfig.cmake)

# Control graphics backend support
set(VCL_OPENGL_SUPPORT CACHE BOOL "Enable OpenGL support")
set(VCL_EGL_SUPPORT CACHE BOOL "Enable EGL support")
set(VCL_D3D12_SUPPORT CACHE BOOL "Enable Direct3D 12 support")
set(VCL_DAWN_WEBGPU_SUPPORT CACHE BOOL "Enable Google Dawn WebGPU support")
set(VCL_WEBGPU_SUPPORT CACHE BOOL "Enable WebGPU support")

# Generate library
vcl_add_library(vcl_graphics)

set(VCL_CONFIG_INC
	vcl/config/opengl.h
	vcl/config/direct3d12.h
	vcl/config/webgpu.h
)

set(SOURCE_OPENGL
	vcl/graphics/imageprocessing/opengl/GaussianBlur.h
	vcl/graphics/imageprocessing/opengl/conversion.cpp
	vcl/graphics/imageprocessing/opengl/conversion.h
	vcl/graphics/imageprocessing/opengl/gaussian.cpp
	vcl/graphics/imageprocessing/opengl/gaussian.h
	vcl/graphics/imageprocessing/opengl/imageprocessor.cpp
	vcl/graphics/imageprocessing/opengl/imageprocessor.h
	vcl/graphics/imageprocessing/opengl/luminance.cpp
	vcl/graphics/imageprocessing/opengl/luminance.h
	vcl/graphics/imageprocessing/opengl/srgb.cpp
	vcl/graphics/imageprocessing/opengl/srgb.h
	vcl/graphics/imageprocessing/opengl/tonemap.cpp
	vcl/graphics/imageprocessing/opengl/tonemap.h
	vcl/graphics/opengl/algorithm/histogram.comp
	vcl/graphics/opengl/algorithm/histogram.cpp
	vcl/graphics/opengl/algorithm/histogram.h
	vcl/graphics/opengl/algorithm/radixsort.comp
	vcl/graphics/opengl/algorithm/radixsort.cpp
	vcl/graphics/opengl/algorithm/radixsort.h
	vcl/graphics/opengl/algorithm/scan.comp
	vcl/graphics/opengl/algorithm/scan.cpp
	vcl/graphics/opengl/algorithm/scan.glslinc
	vcl/graphics/opengl/algorithm/scan.h
	vcl/graphics/opengl/commandstream.cpp
	vcl/graphics/opengl/commandstream.h
	vcl/graphics/opengl/context.cpp
	vcl/graphics/opengl/context.h
	vcl/graphics/opengl/drawcmds.h
	vcl/graphics/opengl/gl.cpp
	vcl/graphics/opengl/gl.h
	vcl/graphics/opengl/glsl/uniformbuffer.h
	vcl/graphics/opengl/type_traits.cpp
	vcl/graphics/opengl/type_traits.h
	vcl/graphics/runtime/opengl/graphicsengine.cpp
	vcl/graphics/runtime/opengl/graphicsengine.h
	vcl/graphics/runtime/opengl/resource/buffer.cpp
	vcl/graphics/runtime/opengl/resource/buffer.h
	vcl/graphics/runtime/opengl/resource/resource.h
	vcl/graphics/runtime/opengl/resource/shader.cpp
	vcl/graphics/runtime/opengl/resource/shader.h
	vcl/graphics/runtime/opengl/resource/texture.cpp
	vcl/graphics/runtime/opengl/resource/texture.h
	vcl/graphics/runtime/opengl/resource/texture1d.cpp
	vcl/graphics/runtime/opengl/resource/texture1d.h
	vcl/graphics/runtime/opengl/resource/texture1darray.cpp
	vcl/graphics/runtime/opengl/resource/texture1darray.h
	vcl/graphics/runtime/opengl/resource/texture2d.cpp
	vcl/graphics/runtime/opengl/resource/texture2d.h
	vcl/graphics/runtime/opengl/resource/texture2darray.cpp
	vcl/graphics/runtime/opengl/resource/texture2darray.h
	vcl/graphics/runtime/opengl/resource/texture3d.cpp
	vcl/graphics/runtime/opengl/resource/texture3d.h
	vcl/graphics/runtime/opengl/resource/texturecube.cpp
	vcl/graphics/runtime/opengl/resource/texturecube.h
	vcl/graphics/runtime/opengl/resource/texturecubearray.cpp
	vcl/graphics/runtime/opengl/resource/texturecubearray.h
	vcl/graphics/runtime/opengl/state/blendstate.cpp
	vcl/graphics/runtime/opengl/state/blendstate.h
	vcl/graphics/runtime/opengl/state/depthstencilstate.cpp
	vcl/graphics/runtime/opengl/state/depthstencilstate.h
	vcl/graphics/runtime/opengl/state/framebuffer.cpp
	vcl/graphics/runtime/opengl/state/framebuffer.h
	vcl/graphics/runtime/opengl/state/inputlayout.cpp
	vcl/graphics/runtime/opengl/state/inputlayout.h
	vcl/graphics/runtime/opengl/state/pipelinestate.cpp
	vcl/graphics/runtime/opengl/state/pipelinestate.h
	vcl/graphics/runtime/opengl/state/rasterizerstate.cpp
	vcl/graphics/runtime/opengl/state/rasterizerstate.h
	vcl/graphics/runtime/opengl/state/sampler.cpp
	vcl/graphics/runtime/opengl/state/sampler.h
	vcl/graphics/runtime/opengl/state/shaderprogram.cpp
	vcl/graphics/runtime/opengl/state/shaderprogram.h
)

set(SOURCE_D3D12
	vcl/graphics/d3d12/commandqueue.cpp
	vcl/graphics/d3d12/commandqueue.h
	vcl/graphics/d3d12/d3d.cpp
	vcl/graphics/d3d12/d3d.h
	vcl/graphics/d3d12/descriptortable.cpp
	vcl/graphics/d3d12/descriptortable.h
	vcl/graphics/d3d12/device.cpp
	vcl/graphics/d3d12/device.h
	vcl/graphics/d3d12/semaphore.cpp
	vcl/graphics/d3d12/semaphore.h
	vcl/graphics/d3d12/swapchain.cpp
	vcl/graphics/d3d12/swapchain.h
	vcl/graphics/runtime/d3d12/graphicsengine.cpp
	vcl/graphics/runtime/d3d12/graphicsengine.h
	vcl/graphics/runtime/d3d12/resource/buffer.cpp
	vcl/graphics/runtime/d3d12/resource/buffer.h
	vcl/graphics/runtime/d3d12/resource/shader.cpp
	vcl/graphics/runtime/d3d12/resource/shader.h
	vcl/graphics/runtime/d3d12/resource/texture.cpp
	vcl/graphics/runtime/d3d12/resource/texture.h
	vcl/graphics/runtime/d3d12/state/blendstate.cpp
	vcl/graphics/runtime/d3d12/state/blendstate.h
	vcl/graphics/runtime/d3d12/state/depthstencilstate.cpp
	vcl/graphics/runtime/d3d12/state/depthstencilstate.h
	vcl/graphics/runtime/d3d12/state/inputlayout.cpp
	vcl/graphics/runtime/d3d12/state/inputlayout.h
	vcl/graphics/runtime/d3d12/state/pipelinestate.cpp
	vcl/graphics/runtime/d3d12/state/pipelinestate.h
	vcl/graphics/runtime/d3d12/state/rasterizerstate.cpp
	vcl/graphics/runtime/d3d12/state/rasterizerstate.h
)

set(SOURCE_WEBGPU
	vcl/graphics/runtime/webgpu/resource/buffer.cpp
	vcl/graphics/runtime/webgpu/resource/buffer.h
	vcl/graphics/runtime/webgpu/resource/shader.cpp
	vcl/graphics/runtime/webgpu/resource/shader.h
	vcl/graphics/runtime/webgpu/state/blendstate.cpp
	vcl/graphics/runtime/webgpu/state/blendstate.h
	vcl/graphics/runtime/webgpu/state/depthstencilstate.cpp
	vcl/graphics/runtime/webgpu/state/depthstencilstate.h
	vcl/graphics/runtime/webgpu/state/inputlayout.cpp
	vcl/graphics/runtime/webgpu/state/inputlayout.h
	vcl/graphics/runtime/webgpu/state/pipelinestate.cpp
	vcl/graphics/runtime/webgpu/state/pipelinestate.h
	vcl/graphics/runtime/webgpu/state/rasterizerstate.cpp
	vcl/graphics/runtime/webgpu/state/rasterizerstate.h
	vcl/graphics/webgpu/helpers.h
	vcl/graphics/webgpu/swapchain.cpp
	vcl/graphics/webgpu/swapchain.h
	vcl/graphics/webgpu/webgpu.cpp
	vcl/graphics/webgpu/webgpu.h
)

set(SOURCE
	vcl/graphics/imageprocessing/conversion.cpp
	vcl/graphics/imageprocessing/conversion.h
	vcl/graphics/imageprocessing/gaussian.cpp
	vcl/graphics/imageprocessing/gaussian.h
	vcl/graphics/imageprocessing/imageprocessor.cpp
	vcl/graphics/imageprocessing/imageprocessor.h
	vcl/graphics/imageprocessing/link.cpp
	vcl/graphics/imageprocessing/link.h
	vcl/graphics/imageprocessing/luminance.cpp
	vcl/graphics/imageprocessing/luminance.h
	vcl/graphics/imageprocessing/srgb.cpp
	vcl/graphics/imageprocessing/srgb.h
	vcl/graphics/imageprocessing/task.cpp
	vcl/graphics/imageprocessing/task.h
	vcl/graphics/imageprocessing/tonemap.cpp
	vcl/graphics/imageprocessing/tonemap.h

	vcl/graphics/runtime/framebuffer.cpp
	vcl/graphics/runtime/framebuffer.h
	vcl/graphics/runtime/graphicsengine.h
	vcl/graphics/runtime/resource/buffer.cpp
	vcl/graphics/runtime/resource/buffer.h
	vcl/graphics/runtime/resource/shader.cpp
	vcl/graphics/runtime/resource/shader.h
	vcl/graphics/runtime/resource/texture.cpp
	vcl/graphics/runtime/resource/texture.h
	vcl/graphics/runtime/state/blendstate.h
	vcl/graphics/runtime/state/common.h
	vcl/graphics/runtime/state/depthstencilstate.h
	vcl/graphics/runtime/state/inputlayout.h
	vcl/graphics/runtime/state/pipelinestate.h
	vcl/graphics/runtime/state/rasterizerstate.h
	vcl/graphics/runtime/state/sampler.cpp
	vcl/graphics/runtime/state/sampler.h

	vcl/graphics/shadows/algorithms.h
	vcl/graphics/shadows/shadowmap.cpp
	vcl/graphics/shadows/shadowmap.h
	vcl/graphics/shadows/shadowvolume.cpp
	vcl/graphics/shadows/shadowvolume.h
	
	vcl/graphics/camera.cpp
	vcl/graphics/camera.h
	vcl/graphics/cameracontroller.h
	vcl/graphics/frustum.cpp
	vcl/graphics/frustum.h
	vcl/graphics/matrixfactory.cpp
	vcl/graphics/matrixfactory.h
	vcl/graphics/surfaceformat.cpp
	vcl/graphics/surfaceformat.h
	vcl/graphics/trackball.cpp
	vcl/graphics/trackball.h
	vcl/graphics/trackballcameracontroller.cpp
	vcl/graphics/trackballcameracontroller.h
)

vcl_target_sources(vcl_graphics "vcl"  ${VCL_CONFIG_INC})
vcl_target_sources(vcl_graphics "vcl/graphics" ${SOURCE})
target_link_libraries(vcl_graphics
	PUBLIC
		vcl_core
)

if(VCL_OPENGL_SUPPORT)
	find_package(OpenGL REQUIRED)

	vcl_target_sources(vcl_graphics "vcl/graphics" ${SOURCE_OPENGL})
	
	target_compile_definitions(vcl_graphics PUBLIC VCL_OPENGL_SUPPORT)

	if(NOT EMSCRIPTEN)
		target_link_libraries(vcl_graphics
			PUBLIC
				OpenGL::GL
		)

		if (NOT vcl_ext_glew)
			find_package(GLEW QUIET)
			if(GLEW_FOUND)
				set(vcl_ext_glew GLEW::glew)
			else()
				set(vcl_ext_glew)
			endif()
		endif()

		if(TARGET ${vcl_ext_glew})
			target_link_libraries(vcl_graphics
				PUBLIC
					${vcl_ext_glew}
			)
		elseif(WIN32)
			FetchContent_Declare(
				glew
				URL https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0-win32.zip
			)
			FetchContent_GetProperties(glew)
			if(NOT glew_POPULATED)
				FetchContent_Populate(glew)
				message(STATUS "Downloaded GLEW to ${glew_SOURCE_DIR}")
			endif()

			set(GLEW_DIR "${glew_SOURCE_DIR}")
			set(GLEW_INCLUDE_DIR "${glew_SOURCE_DIR}/include")
			set(GLEW_SHARED_LIBRARY_DEBUG "${glew_SOURCE_DIR}/lib/Release/x64/glew32.lib")
			set(GLEW_SHARED_LIBRARY_RELEASE "${glew_SOURCE_DIR}/lib/Release/x64/glew32.lib")
			add_custom_command(
				TARGET ${PROJECT_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					${glew_SOURCE_DIR}/bin/Release/x64/glew32.dll
					$<TARGET_FILE_DIR:${PROJECT_NAME}>)

			find_package(GLEW QUIET)
			target_link_libraries(vcl_graphics
				PUBLIC
					GLEW::glew
			)
		endif()
	
		if(VCL_EGL_SUPPORT AND TARGET ${vcl_ext_egl})
			target_compile_definitions(vcl_graphics PUBLIC VCL_EGL_SUPPORT)
			target_link_libraries(vcl_graphics
				PUBLIC
					${vcl_ext_egl}
			)
		endif()
	endif()
endif()

if(VCL_D3D12_SUPPORT)
	vcl_target_sources(vcl_graphics "vcl/graphics" ${SOURCE_D3D12})

	if(VCL_COMPILER_MSVC AND
	   ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER "19.11" AND
	   "${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}" VERSION_LESS "10.0.18362.0"
	)
		message(WARNING "Enable two-phase lookup for templates due to incompatible WindowsSDK")
		target_compile_options(vcl_graphics PUBLIC "/Zc:twoPhase-")
	endif()

	target_compile_definitions(vcl_graphics PUBLIC VCL_D3D12_SUPPORT)
	target_link_libraries(vcl_graphics
		PUBLIC
			absl_variant
			d3d12.lib dxgi.lib d3dcompiler.lib dxguid.lib
	)
endif()

if(VCL_WEBGPU_SUPPORT AND NOT EMSCRIPTEN)
	if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
		message(WARNING "Downloading Google Dawn automatically requires CMake 3.11+")

		set(DAWN_INCLUDE_DIR CACHE PATH "Dawn WebGPU include path")
		set(DAWN_NATIVE_LIBRARY CACHE FILEPATH "Dawn WebGPU library")
		set(DAWN_PROC_LIBRARY CACHE FILEPATH "Dawn WebGPU procedure hooks library")

		set(WEBGPU_INCLUDE_DIR CACHE PATH "WebGPU include path")
		set(WEBGPU_CPP_LIBRARY CACHE FILEPATH "WebGPU C++ library")
	else()
		if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
			FetchContent_Declare(
				webgpu_library
				URL      https://github.com/bfierz/dawn-builds/releases/download/0.0.20210122-72cd1a5e/dawn_win_x64_debug.zip
			)
		elseif("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Linux")
			#FetchContent_Declare(
			#	webgpu_library
			#	URL      https://github.com/bfierz/dawn-builds/releases/download/0.0.20210122-72cd1a5e/dawn_win_x64_debug.zip
			#)
		endif()
		FetchContent_GetProperties(webgpu_library)
		if(NOT webgpu_library_POPULATED)
			FetchContent_Populate(webgpu_library)
			message(STATUS "Downloaded Google Dawn to ${webgpu_library_SOURCE_DIR}")

			set(DAWN_INCLUDE_DIR "${webgpu_library_SOURCE_DIR}/include" CACHE PATH "Dawn WebGPU include path")
			set(DAWN_NATIVE_LIBRARY "${webgpu_library_SOURCE_DIR}/lib/dawn_native.dll.lib" CACHE FILEPATH "Dawn WebGPU library")
			set(DAWN_PROC_LIBRARY "${webgpu_library_SOURCE_DIR}/lib/dawn_proc.dll.lib" CACHE FILEPATH "Dawn WebGPU procedure hooks library")

			set(WEBGPU_INCLUDE_DIR "${webgpu_library_SOURCE_DIR}/include" CACHE PATH "WebGPU include path")
			set(WEBGPU_CPP_LIBRARY "${webgpu_library_SOURCE_DIR}/lib/webgpu_cpp.lib" CACHE FILEPATH "WebGPU C++ library")

			add_custom_command(
				TARGET ${PROJECT_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					${webgpu_library_SOURCE_DIR}/bin/dawn_native.dll
					$<TARGET_FILE_DIR:${PROJECT_NAME}>)
			add_custom_command(
				TARGET ${PROJECT_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					${webgpu_library_SOURCE_DIR}/bin/dawn_platform.dll
					$<TARGET_FILE_DIR:${PROJECT_NAME}>)
			add_custom_command(
				TARGET ${PROJECT_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					${webgpu_library_SOURCE_DIR}/bin/dawn_proc.dll
					$<TARGET_FILE_DIR:${PROJECT_NAME}>)
		endif()
	endif()

	vcl_target_sources(vcl_graphics "vcl/graphics" ${SOURCE_WEBGPU})
	target_compile_definitions(vcl_graphics PUBLIC VCL_WEBGPU_SUPPORT)
	target_include_directories(vcl_graphics PUBLIC ${WEBGPU_INCLUDE_DIR} ${DAWN_INCLUDE_DIR})
	target_link_libraries(vcl_graphics
		PUBLIC
			${DAWN_NATIVE_LIBRARY}
			${DAWN_PROC_LIBRARY}
			${WEBGPU_CPP_LIBRARY}
	)
elseif(VCL_WEBGPU_SUPPORT)
	vcl_target_sources(vcl_graphics "vcl/graphics" ${SOURCE_WEBGPU})
	target_compile_definitions(vcl_graphics PUBLIC VCL_WEBGPU_SUPPORT)
endif()

# Setup installation
set(INCLUDE_DIR_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(LIBRARY_DIR_PREFIX "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY vcl DESTINATION ${INCLUDE_DIR_PREFIX})
install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION ${LIBRARY_DIR_PREFIX})
