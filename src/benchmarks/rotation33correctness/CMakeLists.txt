PROJECT(rotation33correctness)

# Status message
MESSAGE(STATUS "Configuring 'rotation33correctness'")

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

ADD_EXECUTABLE(rotation33correctness ${SRC})
SET_TARGET_PROPERTIES(rotation33correctness PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(rotation33correctness
	vcl_core
	vcl_math
)
