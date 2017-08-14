PROJECT(glgridcube)

# Status message
MESSAGE(STATUS "Configuring 'glgridcube'")

# NanoGUI
SET(VCL_NANOGUI_INCLUDE CACHE PATH "Include directory of NanoGUI")
SET(VCL_NANOGUI_DEBUG_LIBRARY CACHE FILEPATH "NanoGUI library")
SET(VCL_NANOGUI_RELEASE_LIBRARY CACHE FILEPATH "NanoGUI library")

# Find includes in corresponding build directories
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

SET(INC
)

SET(SRC
	main.cpp
)

SET(SHADERS
	shaders/boundinggrid.vert
	shaders/boundinggrid.geom
	shaders/boundinggrid.frag
)

SOURCE_GROUP("shaders" FILES ${SHADERS})
SOURCE_GROUP("" FILES ${SRC} ${INC})

# Include dependencies
INCLUDE_DIRECTORIES(${VCL_NANOGUI_INCLUDE})

ADD_EXECUTABLE(glgridcube ${SRC} ${INC} ${SHADERS})
SET_TARGET_PROPERTIES(glgridcube PROPERTIES FOLDER examples/OpenGL)

TARGET_LINK_LIBRARIES(glgridcube
	vcl_graphics
	debug ${VCL_NANOGUI_DEBUG_LIBRARY}
	optimized ${VCL_NANOGUI_RELEASE_LIBRARY}
)
