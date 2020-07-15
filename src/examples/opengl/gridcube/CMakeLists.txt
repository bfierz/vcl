PROJECT(glgridcube)

# Status message
MESSAGE(STATUS "Configuring 'glgridcube'")

# GLSL compiler
INCLUDE(${CMAKE_SOURCE_DIR}/src/cmake/VCLCompileGLSL.cmake)

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
	shaders/boundinggrid.h
	shaders/boundinggrid.vert
	shaders/boundinggrid.geom
	shaders/boundinggrid.frag
)

SOURCE_GROUP("shaders" FILES ${SHADERS})
SOURCE_GROUP("" FILES ${SRC} ${INC})

# Access the include directories in order to compile the GLSL shaders
GET_PROPERTY(CURR_INC_DIRS_0 TARGET vcl_graphics PROPERTY INCLUDE_DIRECTORIES)
LIST(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CURR_INC_DIRS_0})

VCLCOMPILEGLSL(
	${PROJECT_SOURCE_DIR}/shaders/boundinggrid.vert
	"opengl"
	"BoundingGridVert"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_0
)
VCLCOMPILEGLSL(
	${PROJECT_SOURCE_DIR}/shaders/boundinggrid.geom
	"opengl"
	"BoundingGridGeom"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_1
)
VCLCOMPILEGLSL(
	${PROJECT_SOURCE_DIR}/shaders/boundinggrid.frag
	"opengl"
	"BoundingGridFrag"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_2
)
SET(COMPILEDSHADERS ${COMPILEDSHADERS_0} ${COMPILEDSHADERS_1} ${COMPILEDSHADERS_2})

# Include dependencies
INCLUDE_DIRECTORIES(${VCL_NANOGUI_INCLUDE})

ADD_EXECUTABLE(glgridcube ${SRC} ${INC} ${SHADERS} ${COMPILEDSHADERS})
SET_TARGET_PROPERTIES(glgridcube PROPERTIES FOLDER examples/OpenGL)

TARGET_LINK_LIBRARIES(glgridcube
	vcl_graphics
	debug ${VCL_NANOGUI_DEBUG_LIBRARY}
	optimized ${VCL_NANOGUI_RELEASE_LIBRARY}
)
