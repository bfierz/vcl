PROJECT(vkclearscreen)

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

INCLUDE_DIRECTORIES(../../glfw/include)

ADD_EXECUTABLE(vkclearscreen ${SRC})
SET_TARGET_PROPERTIES(vkclearscreen PROPERTIES FOLDER examples/vulkan)

TARGET_LINK_LIBRARIES(vkclearscreen
	vcl_graphics
	glfw
)
