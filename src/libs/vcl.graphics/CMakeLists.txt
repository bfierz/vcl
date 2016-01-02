PROJECT(vcl_graphics)

# VCL / CONFIG
SET(VCL_CONFIG_INC
	vcl/config/opengl.h
)

# VCL / GRAPHICS / OPENGL
SET(VCL_GRAPHICS_OPENGL_INC
	vcl/graphics/opengl/gl.h
	vcl/graphics/opengl/type_traits.h
)
SET(VCL_GRAPHICS_OPENGL_SRC
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
	vcl/graphics/runtime/opengl/state/sampler.h
	vcl/graphics/runtime/opengl/state/shaderprogram.h
)
SET(VCL_GRAPHICS_RUNTIME_OPENGL_STATE_SRC
	vcl/graphics/runtime/opengl/state/blendstate.cpp
	vcl/graphics/runtime/opengl/state/framebuffer.cpp
	vcl/graphics/runtime/opengl/state/inputlayout.cpp
	vcl/graphics/runtime/opengl/state/sampler.cpp
	vcl/graphics/runtime/opengl/state/shaderprogram.cpp
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
	vcl/graphics/runtime/state/inputlayout.h
	vcl/graphics/runtime/state/sampler.h
)
SET(VCL_GRAPHICS_RUNTIME_STATE_SRC
	vcl/graphics/runtime/state/sampler.cpp
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
SOURCE_GROUP("opengl" FILES ${VCL_GRAPHICS_OPENGL_SRC} ${VCL_GRAPHICS_OPENGL_INC})
SOURCE_GROUP("runtime\\opengl\\resource" FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_INC})
SOURCE_GROUP("runtime\\opengl\\state" FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_INC})
SOURCE_GROUP("runtime\\resource" FILES ${VCL_GRAPHICS_RUNTIME_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_RESOURCE_INC})
SOURCE_GROUP("runtime\\state" FILES ${VCL_GRAPHICS_RUNTIME_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_STATE_INC})
SOURCE_GROUP("shadows" FILES ${VCL_GRAPHICS_SHADOWS_SRC} ${VCL_GRAPHICS_SHADOWS_INC})
SOURCE_GROUP("" FILES ${VCL_GRAPHICS_SRC} ${VCL_GRAPHICS_INC})

SET(SOURCE
	${VCL_CONFIG_INC}
	
	${VCL_GRAPHICS_OPENGL_SRC} ${VCL_GRAPHICS_OPENGL_INC}
	
	${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_INC}
	${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_INC}
	${VCL_GRAPHICS_RUNTIME_RESOURCE_SRC} ${VCL_GRAPHICS_RUNTIME_RESOURCE_INC}
	${VCL_GRAPHICS_RUNTIME_STATE_SRC} ${VCL_GRAPHICS_RUNTIME_STATE_INC}

	${VCL_GRAPHICS_SHADOWS_SRC} ${VCL_GRAPHICS_SHADOWS_INC}
	${VCL_GRAPHICS_SRC} ${VCL_GRAPHICS_INC}
)

# Generate library
ADD_LIBRARY(vcl_graphics STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_graphics PROPERTIES FOLDER libs)
SET_TARGET_PROPERTIES(vcl_graphics PROPERTIES DEBUG_POSTFIX _d)
TARGET_INCLUDE_DIRECTORIES(vcl_graphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${GLEW_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(vcl_graphics
	vcl_core
	${GLEW_LIBRARY}
	${OPENGL_gl_LIBRARY}
)

# Setup installation
INSTALL(FILES ${VCL_CONFIG_INC} DESTINATION include/vcl/config)
INSTALL(FILES ${VCL_GRAPHICS_OPENGL_INC} DESTINATION include/vcl/graphics/opengl)
INSTALL(FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_RESOURCE_INC} DESTINATION include/vcl/graphics/runtime/opengl/resource)
INSTALL(FILES ${VCL_GRAPHICS_RUNTIME_OPENGL_STATE_INC} DESTINATION include/vcl/graphics/runtime/opengl/state)
INSTALL(FILES ${VCL_GRAPHICS_RUNTIME_RESOURCE_INC} DESTINATION include/vcl/graphics/runtime/resource)
INSTALL(FILES ${VCL_GRAPHICS_RUNTIME_STATE_INC} DESTINATION include/vcl/graphics/runtime/state)
INSTALL(FILES ${VCL_GRAPHICS_SHADOWS_INC} DESTINATION include/vcl/graphics/shadows)
INSTALL(FILES ${VCL_GRAPHICS_INC} DESTINATION include/vcl/graphics)
INSTALL(TARGETS vcl_graphics ARCHIVE DESTINATION lib)
