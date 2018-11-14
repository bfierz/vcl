PROJECT(rtwturb)

# Status message
MESSAGE(STATUS "Configuring 'rtwturb'")

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
	shaders/smoke.vert
	shaders/smoke.geom
	shaders/smoke.frag
)

SOURCE_GROUP("shaders" FILES ${SHADERS})
SOURCE_GROUP("" FILES ${SRC} ${INC})

# Include dependencies
INCLUDE_DIRECTORIES(${VCL_NANOGUI_INCLUDE})

ADD_EXECUTABLE(rtwturb ${SRC} ${INC} ${SHADERS})
SET_TARGET_PROPERTIES(rtwturb PROPERTIES FOLDER Examples)

TARGET_LINK_LIBRARIES(rtwturb
	vcl_physics_fluid
	vcl_graphics
	debug ${VCL_NANOGUI_DEBUG_LIBRARY}
	optimized ${VCL_NANOGUI_RELEASE_LIBRARY}
)
