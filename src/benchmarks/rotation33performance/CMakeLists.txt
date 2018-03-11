PROJECT(rotation33performance)

# Status message
MESSAGE(STATUS "Configuring 'rotation33performance'")

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

ADD_EXECUTABLE(rotation33performance ${SRC})
SET_TARGET_PROPERTIES(rotation33performance PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(rotation33performance
	vcl_core
	vcl_math
	benchmark
)
