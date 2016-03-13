PROJECT(vcl_graphics)

# VCL / CONFIG
SET(VCL_CONFIG_INC
	vcl/config/opengl.h
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
	vcl/graphics/runtime/opengl/state/framebuffer.h
	vcl/graphics/runtime/opengl/state/inputlayout.h
	vcl/graphics/runtime/opengl/state/pipelinestate.h
	vcl/graphics/runtime/opengl/state/rasterizerstate.h
	vcl/graphics/runtime/opengl/state/sampler.h
	vcl/graphics/runtime/opengl/state/shaderprogram.h
)
SET(VCL_GRAPHICS_RUNTIME_OPENGL_STATE_SRC
	vcl/graphics/runtime/opengl/state/blendstate.cpp
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

# VCL / GRAPHICS / RUNTIME / VULKAN / RESOURCE
SET(VCL_GRAPHICS_RUNTIME_VULKAN_RESOURCE_INC
	vcl/graphics/runtime/vulkan/resource/buffer.h
	vcl/graphics/runtime/vulkan/resource/shader.h
)
SET(VCL_GRAPHICS_RUNTIME_VULKAN_RESOURCE_SRC
	vcl/graphics/runtime/vulkan/resource/buffer.cpp
	vcl/graphics/runtime/vulkan/resource/shader.cpp
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

# VCL / GRAPHICS / VULKAN
SET(VCL_GRAPHICS_VULKAN_INC
	vcl/graphics/vulkan/commands.h
	vcl/graphics/vulkan/context.h
	vcl/graphics/vulkan/device.h
	vcl/graphics/vulkan/memory.h
	vcl/graphics/vulkan/pipelinestate.h
	vcl/graphics/vulkan/platform.h
	vcl/graphics/vulkan/swapchain.h
	vcl/graphics/vulkan/tools.h
)
SET(VCL_GRAPHICS_VULKAN_SRC
	vcl/graphics/vulkan/commands.cpp
	vcl/graphics/vulkan/context.cpp
	vcl/graphics/vulkan/device.cpp
	vcl/graphics/vulkan/memory.cpp
	vcl/graphics/vulkan/pipelinestate.cpp
	vcl/graphics/vulkan/platform.cpp
	vcl/graphics/vulkan/swapchain.cpp
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
SOURCE_GROUP("opengl\\algorithm" FILES ${VCL_GRAPHICS_OPENGL_ALGORITHM_SRC} ${VCL_GRAPHICS_OPENGL_ALGORITHM_INC})
SOURCE_GROUP("opengl\\glsl" FILES ${VCL_GRAPHICS_OPENGL_GLSL_INC})
SOURCE_GROUP("opengl" FILES ${VCL_GRAPHICS_OPENGL_SRC} ${VCL_GRAPHICS_OPENGL_INC})
SOURCE_GROUP("runtime\\opengl\\resource" FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_INC})
SOURCE_GROUP("runtime\\opengl\\state" FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_INC})
SOURCE_GROUP("runtime\\opengl" FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_INC})
SOURCE_GROUP("runtime\\vulkan\\resource" FILES ${VCL_GRAPHICS_RUNTIME_VULKAN_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_VULKAN_RESOURCE_INC})
SOURCE_GROUP("runtime\\resource" FILES ${VCL_GRAPHICS_RUNTIME_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_RESOURCE_INC})
SOURCE_GROUP("runtime\\state" FILES ${VCL_GRAPHICS_RUNTIME_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_STATE_INC})
SOURCE_GROUP("runtime" FILES ${VCL_GRAPHICS_RUNTIME_SRC} ${VCL_GRAPHICS_RUNTIME_INC})
SOURCE_GROUP("shadows" FILES ${VCL_GRAPHICS_SHADOWS_SRC} ${VCL_GRAPHICS_SHADOWS_INC})
SOURCE_GROUP("vulkan" FILES ${VCL_GRAPHICS_VULKAN_SRC} ${VCL_GRAPHICS_VULKAN_INC})
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

SET(VULKAN_SOURCE
	${VCL_GRAPHICS_RUNTIME_VULKAN_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_VULKAN_RESOURCE_INC}
	
	${VCL_GRAPHICS_VULKAN_SRC} ${VCL_GRAPHICS_VULKAN_INC}
)

# Control OpenGL support
SET(VCL_OPENGL_SUPPORT CACHE BOOL "Enable OpenGL support")

# Control Vulkan support
SET(VCL_VULKAN_SUPPORT CACHE BOOL "Enable Vulkan support")

# Generate library
ADD_LIBRARY(vcl_graphics STATIC ${SOURCE})

IF(VCL_OPENGL_SUPPORT)
	TARGET_SOURCES(vcl_graphics PRIVATE ${OPENGL_SOURCE})
ENDIF()

IF(VCL_VULKAN_SUPPORT)
	FIND_PACKAGE(Vulkan REQUIRED)
	IF(Vulkan_FOUND)
		TARGET_COMPILE_DEFINITIONS(vcl_graphics PUBLIC VCL_VULKAN_SUPPORT)
		TARGET_INCLUDE_DIRECTORIES(vcl_graphics PUBLIC ${Vulkan_INCLUDE_DIR})
		TARGET_LINK_LIBRARIES(vcl_graphics ${Vulkan_LIBRARY})
	ENDIF()
ENDIF()

SET_TARGET_PROPERTIES(vcl_graphics PROPERTIES FOLDER libs)
if(WIN32)
	SET_TARGET_PROPERTIES(vcl_graphics PROPERTIES DEBUG_POSTFIX _d)
endif()
TARGET_INCLUDE_DIRECTORIES(vcl_graphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Check dependency requirements
if(TARGET ${vcl_ext_glew})
	find_package(OpenGL REQUIRED)
	
	target_compile_definitions(vcl_graphics PUBLIC VCL_OPENGL_SUPPORT)
	target_link_libraries(vcl_graphics
		${OPENGL_gl_LIBRARY}
		${vcl_ext_glew}
	)
	if(TARGET ${vcl_ext_egl})
		target_compile_definitions(vcl_graphics PUBLIC VCL_EGL_SUPPORT)
		target_link_libraries(vcl_graphics
			${vcl_ext_egl}
		)
	endif()
endif()

TARGET_LINK_LIBRARIES(vcl_graphics
	vcl_core
)

# Setup installation
INSTALL(FILES ${VCL_CONFIG_INC} DESTINATION include/vcl/config)
INSTALL(FILES ${VCL_GRAPHICS_IMAGEPROCESSING_INC} DESTINATION include/vcl/graphics/imageprocessing)
INSTALL(FILES ${VCL_GRAPHICS_OPENGL_ALGORITHM_INC} DESTINATION include/vcl/graphics/opengl/algorithm)
INSTALL(FILES ${VCL_GRAPHICS_OPENGL_ALGORITHM_INC} DESTINATION include/vcl/graphics/opengl/glsl)
INSTALL(FILES ${VCL_GRAPHICS_RUNTIME_VULKAN_RESOURCE_INC} DESTINATION include/vcl/graphics/runtime/vulkan/resource)
INSTALL(FILES ${VCL_GRAPHICS_RUNTIME_RESOURCE_INC} DESTINATION include/vcl/graphics/runtime/resource)
INSTALL(FILES ${VCL_GRAPHICS_RUNTIME_STATE_INC} DESTINATION include/vcl/graphics/runtime/state)
INSTALL(FILES ${VCL_GRAPHICS_SHADOWS_INC} DESTINATION include/vcl/graphics/shadows)
INSTALL(FILES ${VCL_GRAPHICS_VULKAN_INC} DESTINATION include/vcl/graphics/vulkan)
INSTALL(FILES ${VCL_GRAPHICS_INC} DESTINATION include/vcl/graphics)

IF(VCL_OPENGL_SUPPORT)
	INSTALL(FILES ${VCL_GRAPHICS_IMAGEPROCESSING_OPENGL_INC} DESTINATION include/vcl/graphics/imageprocessing/opengl)
	INSTALL(FILES ${VCL_GRAPHICS_OPENGL_INC} DESTINATION include/vcl/graphics/opengl)
	INSTALL(FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_INC} DESTINATION include/vcl/graphics/runtime/opengl/resource)
	INSTALL(FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_INC} DESTINATION include/vcl/graphics/runtime/opengl/state)
ENDIF()

INSTALL(TARGETS vcl_graphics ARCHIVE DESTINATION lib)
