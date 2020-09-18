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
PROJECT(vcl_graphics)

# VCL / CONFIG
SET(VCL_CONFIG_INC
	vcl/config/opengl.h
	vcl/config/direct3d12.h
)

# VCL / GRAPHICS / IMAGEPROCESSING / OPENGL
SET(VCL_GRAPHICS_IMAGEPROCESSING_OPENGL_INC
	vcl/graphics/imageprocessing/opengl/conversion.h
	vcl/graphics/imageprocessing/opengl/GaussianBlur.h
	vcl/graphics/imageprocessing/opengl/gaussian.h
	vcl/graphics/imageprocessing/opengl/imageprocessor.h
	vcl/graphics/imageprocessing/opengl/luminance.h
	vcl/graphics/imageprocessing/opengl/srgb.h
	vcl/graphics/imageprocessing/opengl/tonemap.h
)
SET(VCL_GRAPHICS_IMAGEPROCESSING_OPENGL_SRC
	vcl/graphics/imageprocessing/opengl/conversion.cpp
	vcl/graphics/imageprocessing/opengl/gaussian.cpp
	vcl/graphics/imageprocessing/opengl/imageprocessor.cpp
	vcl/graphics/imageprocessing/opengl/luminance.cpp
	vcl/graphics/imageprocessing/opengl/srgb.cpp
	vcl/graphics/imageprocessing/opengl/tonemap.cpp
)

# VCL / GRAPHICS / IMAGEPROCESSING
SET(VCL_GRAPHICS_IMAGEPROCESSING_INC
	vcl/graphics/imageprocessing/conversion.h
	vcl/graphics/imageprocessing/gaussian.h
	vcl/graphics/imageprocessing/imageprocessor.h
	vcl/graphics/imageprocessing/link.h
	vcl/graphics/imageprocessing/luminance.h
	vcl/graphics/imageprocessing/srgb.h
	vcl/graphics/imageprocessing/task.h
	vcl/graphics/imageprocessing/tonemap.h
)
SET(VCL_GRAPHICS_IMAGEPROCESSING_SRC
	vcl/graphics/imageprocessing/conversion.cpp
	vcl/graphics/imageprocessing/gaussian.cpp
	vcl/graphics/imageprocessing/imageprocessor.cpp
	vcl/graphics/imageprocessing/link.cpp
	vcl/graphics/imageprocessing/luminance.cpp
	vcl/graphics/imageprocessing/srgb.cpp
	vcl/graphics/imageprocessing/task.cpp
	vcl/graphics/imageprocessing/tonemap.cpp
)

# VCL / GRAPHICS / D3D12
SET(VCL_GRAPHICS_D3D12_INC
	vcl/graphics/d3d12/commandqueue.h
	vcl/graphics/d3d12/d3d.h
	vcl/graphics/d3d12/descriptortable.h
	vcl/graphics/d3d12/device.h
	vcl/graphics/d3d12/semaphore.h
	vcl/graphics/d3d12/swapchain.h
)
SET(VCL_GRAPHICS_D3D12_SRC
	vcl/graphics/d3d12/commandqueue.cpp
	vcl/graphics/d3d12/d3d.cpp
	vcl/graphics/d3d12/descriptortable.cpp
	vcl/graphics/d3d12/device.cpp
	vcl/graphics/d3d12/semaphore.cpp
	vcl/graphics/d3d12/swapchain.cpp
)

# VCL / GRAPHICS / OPENGL / ALGORITHM
FILE(GLOB VCL_GRAPHICS_OPENGL_ALGORITHM_INC
	vcl/graphics/opengl/algorithm/*.h
)
FILE(GLOB VCL_GRAPHICS_OPENGL_ALGORITHM_SRC
	vcl/graphics/opengl/algorithm/*.cpp
	vcl/graphics/opengl/algorithm/*.glslinc
	vcl/graphics/opengl/algorithm/*.comp
)

# VCL / GRAPHICS / OPENGL / GLSL
SET(VCL_GRAPHICS_OPENGL_GLSL_INC
	vcl/graphics/opengl/glsl/uniformbuffer.h
)

# VCL / GRAPHICS / OPENGL
SET(VCL_GRAPHICS_OPENGL_INC
	vcl/graphics/opengl/commandstream.h
	vcl/graphics/opengl/context.h
	vcl/graphics/opengl/drawcmds.h
	vcl/graphics/opengl/gl.h
	vcl/graphics/opengl/type_traits.h
)
SET(VCL_GRAPHICS_OPENGL_SRC
	vcl/graphics/opengl/commandstream.cpp
	vcl/graphics/opengl/context.cpp
	vcl/graphics/opengl/gl.cpp
	vcl/graphics/opengl/type_traits.cpp
)

# VCL / GRAPHICS / WEBGPU
SET(VCL_GRAPHICS_WEBGPU_INC
	vcl/graphics/webgpu/webgpu.h
)
SET(VCL_GRAPHICS_WEBGPU_SRC
	vcl/graphics/webgpu/webgpu.cpp
)

# VCL / GRAPHICS / RUNTIME / D3D12 / RESOURCE
SET(VCL_GRAPHICS_RUNTIME_D3D12_RESOURCE_INC
	vcl/graphics/runtime/d3d12/resource/buffer.h
	vcl/graphics/runtime/d3d12/resource/shader.h
	vcl/graphics/runtime/d3d12/resource/texture.h
)
SET(VCL_GRAPHICS_RUNTIME_D3D12_RESOURCE_SRC
	vcl/graphics/runtime/d3d12/resource/buffer.cpp
	vcl/graphics/runtime/d3d12/resource/shader.cpp
	vcl/graphics/runtime/d3d12/resource/texture.cpp
)

# VCL / GRAPHICS / RUNTIME / D3D12 / STATE
SET(VCL_GRAPHICS_RUNTIME_D3D12_STATE_INC
	vcl/graphics/runtime/d3d12/state/blendstate.h
	vcl/graphics/runtime/d3d12/state/depthstencilstate.h
	vcl/graphics/runtime/d3d12/state/inputlayout.h
	vcl/graphics/runtime/d3d12/state/pipelinestate.h
	vcl/graphics/runtime/d3d12/state/rasterizerstate.h
)
SET(VCL_GRAPHICS_RUNTIME_D3D12_STATE_SRC
	vcl/graphics/runtime/d3d12/state/blendstate.cpp
	vcl/graphics/runtime/d3d12/state/depthstencilstate.cpp
	vcl/graphics/runtime/d3d12/state/inputlayout.cpp
	vcl/graphics/runtime/d3d12/state/pipelinestate.cpp
	vcl/graphics/runtime/d3d12/state/rasterizerstate.cpp
)

# VCL / GRAPHICS / RUNTIME / D3D12
SET(VCL_GRAPHICS_RUNTIME_D3D12_INC
	vcl/graphics/runtime/d3d12/graphicsengine.h
)
SET(VCL_GRAPHICS_RUNTIME_D3D12_SRC
	vcl/graphics/runtime/d3d12/graphicsengine.cpp
)

# VCL / GRAPHICS / RUNTIME / OPENGL / RESOURCE
SET(VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_INC
	vcl/graphics/runtime/opengl/resource/buffer.h
	vcl/graphics/runtime/opengl/resource/resource.h
	vcl/graphics/runtime/opengl/resource/shader.h
	vcl/graphics/runtime/opengl/resource/texture.h
	vcl/graphics/runtime/opengl/resource/texture1d.h
	vcl/graphics/runtime/opengl/resource/texture1darray.h
	vcl/graphics/runtime/opengl/resource/texture2d.h
	vcl/graphics/runtime/opengl/resource/texture2darray.h
	vcl/graphics/runtime/opengl/resource/texture3d.h
	vcl/graphics/runtime/opengl/resource/texturecube.h
	vcl/graphics/runtime/opengl/resource/texturecubearray.h
)
SET(VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_SRC
	vcl/graphics/runtime/opengl/resource/buffer.cpp
	vcl/graphics/runtime/opengl/resource/shader.cpp
	vcl/graphics/runtime/opengl/resource/texture.cpp
	vcl/graphics/runtime/opengl/resource/texture1d.cpp
	vcl/graphics/runtime/opengl/resource/texture1darray.cpp
	vcl/graphics/runtime/opengl/resource/texture2d.cpp
	vcl/graphics/runtime/opengl/resource/texture2darray.cpp
	vcl/graphics/runtime/opengl/resource/texture3d.cpp
	vcl/graphics/runtime/opengl/resource/texturecube.cpp
	vcl/graphics/runtime/opengl/resource/texturecubearray.cpp
)

# VCL / GRAPHICS / RUNTIME / OPENGL / STATE
SET(VCL_GRAPHICS_RUNTIME_OPENGL_STATE_INC
	vcl/graphics/runtime/opengl/state/blendstate.h
	vcl/graphics/runtime/opengl/state/depthstencilstate.h
	vcl/graphics/runtime/opengl/state/framebuffer.h
	vcl/graphics/runtime/opengl/state/inputlayout.h
	vcl/graphics/runtime/opengl/state/pipelinestate.h
	vcl/graphics/runtime/opengl/state/rasterizerstate.h
	vcl/graphics/runtime/opengl/state/sampler.h
	vcl/graphics/runtime/opengl/state/shaderprogram.h
)
SET(VCL_GRAPHICS_RUNTIME_OPENGL_STATE_SRC
	vcl/graphics/runtime/opengl/state/blendstate.cpp
	vcl/graphics/runtime/opengl/state/depthstencilstate.cpp
	vcl/graphics/runtime/opengl/state/framebuffer.cpp
	vcl/graphics/runtime/opengl/state/inputlayout.cpp
	vcl/graphics/runtime/opengl/state/pipelinestate.cpp
	vcl/graphics/runtime/opengl/state/rasterizerstate.cpp
	vcl/graphics/runtime/opengl/state/sampler.cpp
	vcl/graphics/runtime/opengl/state/shaderprogram.cpp
)

# VCL / GRAPHICS / RUNTIME / OPENGL
SET(VCL_GRAPHICS_RUNTIME_OPENGL_INC
	vcl/graphics/runtime/opengl/graphicsengine.h
)
SET(VCL_GRAPHICS_RUNTIME_OPENGL_SRC
	vcl/graphics/runtime/opengl/graphicsengine.cpp
)

# VCL / GRAPHICS / RUNTIME / WEBGPU / RESOURCE
SET(VCL_GRAPHICS_RUNTIME_WEBGPU_RESOURCE_INC
	vcl/graphics/runtime/webgpu/resource/buffer.h
	vcl/graphics/runtime/webgpu/resource/shader.h
)
SET(VCL_GRAPHICS_RUNTIME_WEBGPU_RESOURCE_SRC
	vcl/graphics/runtime/webgpu/resource/buffer.cpp
	vcl/graphics/runtime/webgpu/resource/shader.cpp
)

# VCL / GRAPHICS / RUNTIME / WEBGPU / STATE
SET(VCL_GRAPHICS_RUNTIME_WEBGPU_STATE_INC
	vcl/graphics/runtime/webgpu/state/blendstate.h
	vcl/graphics/runtime/webgpu/state/depthstencilstate.h
	vcl/graphics/runtime/webgpu/state/inputlayout.h
	vcl/graphics/runtime/webgpu/state/pipelinestate.h
	vcl/graphics/runtime/webgpu/state/rasterizerstate.h
)
SET(VCL_GRAPHICS_RUNTIME_WEBGPU_STATE_SRC
	vcl/graphics/runtime/webgpu/state/blendstate.cpp
	vcl/graphics/runtime/webgpu/state/depthstencilstate.cpp
	vcl/graphics/runtime/webgpu/state/inputlayout.cpp
	vcl/graphics/runtime/webgpu/state/pipelinestate.cpp
	vcl/graphics/runtime/webgpu/state/rasterizerstate.cpp
)

# VCL / GRAPHICS / RUNTIME / RESOURCE
SET(VCL_GRAPHICS_RUNTIME_RESOURCE_INC
	vcl/graphics/runtime/resource/buffer.h
	vcl/graphics/runtime/resource/shader.h
	vcl/graphics/runtime/resource/texture.h
)
SET(VCL_GRAPHICS_RUNTIME_RESOURCE_SRC
	vcl/graphics/runtime/resource/buffer.cpp
	vcl/graphics/runtime/resource/shader.cpp
	vcl/graphics/runtime/resource/texture.cpp
)

# VCL / GRAPHICS / RUNTIME / STATE
SET(VCL_GRAPHICS_RUNTIME_STATE_INC
	vcl/graphics/runtime/state/blendstate.h
	vcl/graphics/runtime/state/common.h
	vcl/graphics/runtime/state/depthstencilstate.h
	vcl/graphics/runtime/state/inputlayout.h
	vcl/graphics/runtime/state/pipelinestate.h
	vcl/graphics/runtime/state/rasterizerstate.h
	vcl/graphics/runtime/state/sampler.h
)
SET(VCL_GRAPHICS_RUNTIME_STATE_SRC
	vcl/graphics/runtime/state/sampler.cpp
)

# VCL / GRAPHICS / RUNTIME
SET(VCL_GRAPHICS_RUNTIME_INC
	vcl/graphics/runtime/framebuffer.h
	vcl/graphics/runtime/graphicsengine.h
)
SET(VCL_GRAPHICS_RUNTIME_SRC
	vcl/graphics/runtime/framebuffer.cpp
)

# VCL / GRAPHICS / SHADOWS
SET(VCL_GRAPHICS_SHADOWS_INC
	vcl/graphics/shadows/algorithms.h
	vcl/graphics/shadows/shadowmap.h
	vcl/graphics/shadows/shadowvolume.h
)
SET(VCL_GRAPHICS_SHADOWS_SRC
	vcl/graphics/shadows/shadowmap.cpp
	vcl/graphics/shadows/shadowvolume.cpp
)

# VCL / GRAPHICS
SET(VCL_GRAPHICS_INC
	vcl/graphics/cameracontroller.h
	vcl/graphics/camera.h
	vcl/graphics/frustum.h
	vcl/graphics/matrixfactory.h
	vcl/graphics/surfaceformat.h
	vcl/graphics/trackballcameracontroller.h
	vcl/graphics/trackball.h
)
SET(VCL_GRAPHICS_SRC
	vcl/graphics/camera.cpp
	vcl/graphics/frustum.cpp
	vcl/graphics/matrixfactory.cpp
	vcl/graphics/surfaceformat.cpp
	vcl/graphics/trackballcameracontroller.cpp
	vcl/graphics/trackball.cpp
)

SOURCE_GROUP("config" FILES ${VCL_CONFIG_INC})
SOURCE_GROUP("imageprocessing\\opengl" FILES ${VCL_GRAPHICS_IMAGEPROCESSING_OPENGL_SRC} ${VCL_GRAPHICS_IMAGEPROCESSING_OPENGL_INC})
SOURCE_GROUP("imageprocessing" FILES ${VCL_GRAPHICS_IMAGEPROCESSING_SRC} ${VCL_GRAPHICS_IMAGEPROCESSING_INC})
SOURCE_GROUP("d3d12" FILES ${VCL_GRAPHICS_D3D12_SRC} ${VCL_GRAPHICS_D3D12_INC})
SOURCE_GROUP("opengl\\algorithm" FILES ${VCL_GRAPHICS_OPENGL_ALGORITHM_SRC} ${VCL_GRAPHICS_OPENGL_ALGORITHM_INC})
SOURCE_GROUP("opengl\\glsl" FILES ${VCL_GRAPHICS_OPENGL_GLSL_INC})
SOURCE_GROUP("opengl" FILES ${VCL_GRAPHICS_OPENGL_SRC} ${VCL_GRAPHICS_OPENGL_INC})
SOURCE_GROUP("runtime\\d3d12\\resource" FILES ${VCL_GRAPHICS_RUNTIME_D3D12_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_D3D12_RESOURCE_INC})
SOURCE_GROUP("runtime\\d3d12\\state" FILES ${VCL_GRAPHICS_RUNTIME_D3D12_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_D3D12_STATE_INC})
SOURCE_GROUP("runtime\\d3d12" FILES ${VCL_GRAPHICS_RUNTIME_D3D12_SRC} ${VCL_GRAPHICS_RUNTIME_D3D12_INC})
SOURCE_GROUP("runtime\\opengl\\resource" FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_INC})
SOURCE_GROUP("runtime\\opengl\\state" FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_INC})
SOURCE_GROUP("runtime\\opengl" FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_INC})
SOURCE_GROUP("runtime\\webgpu\\resource" FILES ${VCL_GRAPHICS_RUNTIME_WEBGPU_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_WEBGPU_RESOURCE_INC})
SOURCE_GROUP("runtime\\webgpu\\state" FILES ${VCL_GRAPHICS_RUNTIME_WEBGPU_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_WEBGPU_STATE_INC})
SOURCE_GROUP("runtime\\resource" FILES ${VCL_GRAPHICS_RUNTIME_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_RESOURCE_INC})
SOURCE_GROUP("runtime\\state" FILES ${VCL_GRAPHICS_RUNTIME_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_STATE_INC})
SOURCE_GROUP("runtime" FILES ${VCL_GRAPHICS_RUNTIME_SRC} ${VCL_GRAPHICS_RUNTIME_INC})
SOURCE_GROUP("shadows" FILES ${VCL_GRAPHICS_SHADOWS_SRC} ${VCL_GRAPHICS_SHADOWS_INC})
SOURCE_GROUP("webgpu" FILES ${VCL_GRAPHICS_WEBGPU_SRC} ${VCL_GRAPHICS_WEBGPU_INC})
SOURCE_GROUP("" FILES ${VCL_GRAPHICS_SRC} ${VCL_GRAPHICS_INC})

SET(SOURCE
	${VCL_CONFIG_INC}
	
	${VCL_GRAPHICS_IMAGEPROCESSING_SRC} ${VCL_GRAPHICS_IMAGEPROCESSING_INC}
	${VCL_GRAPHICS_RUNTIME_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_RESOURCE_INC}
	${VCL_GRAPHICS_RUNTIME_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_STATE_INC}
	${VCL_GRAPHICS_RUNTIME_SRC} ${VCL_GRAPHICS_RUNTIME_INC}

	${VCL_GRAPHICS_SHADOWS_SRC} ${VCL_GRAPHICS_SHADOWS_INC}
	${VCL_GRAPHICS_SRC} ${VCL_GRAPHICS_INC}
)

SET(OPENGL_SOURCE

	${VCL_GRAPHICS_IMAGEPROCESSING_OPENGL_SRC} ${VCL_GRAPHICS_IMAGEPROCESSING_OPENGL_INC}
	
	${VCL_GRAPHICS_OPENGL_ALGORITHM_SRC} ${VCL_GRAPHICS_OPENGL_ALGORITHM_INC}
	${VCL_GRAPHICS_OPENGL_GLSL_INC}
	${VCL_GRAPHICS_OPENGL_SRC} ${VCL_GRAPHICS_OPENGL_INC}
	
	${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_INC}
	${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_INC}
	${VCL_GRAPHICS_RUNTIME_OPENGL_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_INC}
)

SET(D3D12_SOURCE
	${VCL_GRAPHICS_D3D12_SRC} ${VCL_GRAPHICS_D3D12_INC}

	${VCL_GRAPHICS_RUNTIME_D3D12_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_D3D12_RESOURCE_INC}
	${VCL_GRAPHICS_RUNTIME_D3D12_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_D3D12_STATE_INC}
	${VCL_GRAPHICS_RUNTIME_D3D12_SRC} ${VCL_GRAPHICS_RUNTIME_D3D12_INC}
)

SET(WEBGPU_SOURCE
	${VCL_GRAPHICS_RUNTIME_WEBGPU_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_WEBGPU_RESOURCE_INC}
	${VCL_GRAPHICS_RUNTIME_WEBGPU_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_WEBGPU_STATE_INC}
	${VCL_GRAPHICS_WEBGPU_SRC} ${VCL_GRAPHICS_WEBGPU_INC}
)

# Control graphics backend support
SET(VCL_OPENGL_SUPPORT CACHE BOOL "Enable OpenGL support")
SET(VCL_EGL_SUPPORT CACHE BOOL "Enable EGL support")
SET(VCL_D3D12_SUPPORT CACHE BOOL "Enable Direct3D 12 support")

SET(VCL_DAWN_WEBGPU_SUPPORT CACHE BOOL "Enable Google Dawn WebGPU support")
SET(VCL_WEBGPU_SUPPORT CACHE BOOL "Enable WebGPU support")

# Generate library
ADD_LIBRARY(vcl_graphics STATIC ${SOURCE})

IF(VCL_OPENGL_SUPPORT)
	TARGET_SOURCES(vcl_graphics PRIVATE ${OPENGL_SOURCE})
ENDIF()

if(VCL_D3D12_SUPPORT)
	target_sources(vcl_graphics PRIVATE ${D3D12_SOURCE})

	target_compile_definitions(vcl_graphics PUBLIC VCL_D3D12_SUPPORT)
	target_link_libraries(vcl_graphics
		absl_variant
		d3d12.lib dxgi.lib d3dcompiler.lib dxguid.lib
	)
endif()

if(VCL_WEBGPU_SUPPORT)
	set(DAWN_INCLUDE_DIR CACHE PATH "Dawn WebGPU include path")
	set(DAWN_LIBRARY CACHE FILEPATH "Dawn WebGPU library")

	set(WEBGPU_INCLUDE_DIR CACHE PATH "WebGPU include path")
	set(WEBGPU_LIBRARY CACHE FILEPATH "WebGPU library")

	target_sources(vcl_graphics PRIVATE ${WEBGPU_SOURCE})
	target_compile_definitions(vcl_graphics PUBLIC VCL_WEBGPU_SUPPORT)
	target_include_directories(vcl_graphics PUBLIC ${WEBGPU_INCLUDE_DIR} ${DAWN_INCLUDE_DIR})
	target_link_libraries(vcl_graphics
		${WEBGPU_LIBRARY}
		${DAWN_LIBRARY}
	)
endif()

SET_TARGET_PROPERTIES(vcl_graphics PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_graphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Check dependency requirements
if(TARGET ${vcl_ext_glew})
	find_package(OpenGL REQUIRED)
	
	target_compile_definitions(vcl_graphics PUBLIC VCL_OPENGL_SUPPORT)
	target_link_libraries(vcl_graphics
		${OPENGL_gl_LIBRARY}
		${vcl_ext_glew}
	)
	if(VCL_EGL_SUPPORT AND TARGET ${vcl_ext_egl})
		target_compile_definitions(vcl_graphics PUBLIC VCL_EGL_SUPPORT)
		target_link_libraries(vcl_graphics
			${vcl_ext_egl}
		)
	endif()
elseif(EMSCRIPTEN)
	target_compile_definitions(vcl_graphics PUBLIC VCL_OPENGL_SUPPORT)
endif()

TARGET_LINK_LIBRARIES(vcl_graphics
	vcl_core
)

# Setup installation
set(INCLUDE_DIR_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(LIBRARY_DIR_PREFIX "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY vcl DESTINATION ${INCLUDE_DIR_PREFIX})
install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION ${LIBRARY_DIR_PREFIX})
