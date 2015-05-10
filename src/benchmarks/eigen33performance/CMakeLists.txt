PROJECT(eigen33performance)

# Status message
MESSAGE(STATUS "Configuring 'eigen33performance'")

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

ADD_EXECUTABLE(eigen33performance ${SRC})
SET_TARGET_PROPERTIES(eigen33performance PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(eigen33performance
	vcl_core
	vcl_math
)
