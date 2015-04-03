PROJECT(vcl_graphics)

# VCL / GRAPHICS
SET(VCL_GRAPHICS_INC
	vcl/graphics/camera.h
	vcl/graphics/frustum.h
	vcl/graphics/matrixfactory.h
)
SET(VCL_GRAPHICS_SRC
	vcl/graphics/camera.cpp
	vcl/graphics/frustum.cpp
	vcl/graphics/matrixfactory.cpp
)

SOURCE_GROUP("" FILES ${VCL_GRAPHICS_SRC} ${VCL_GRAPHICS_INC})

SET(SOURCE
	${VCL_GRAPHICS_SRC} ${VCL_GRAPHICS_INC}
)

# Generate library
ADD_LIBRARY(vcl_graphics STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_graphics PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_graphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

TARGET_LINK_LIBRARIES(vcl_graphics
	vcl_core
)