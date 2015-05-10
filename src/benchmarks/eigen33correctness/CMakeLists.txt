PROJECT(eigen33correctness)

# Status message
MESSAGE(STATUS "Configuring 'eigen33correctness'")

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

ADD_EXECUTABLE(eigen33correctness ${SRC})
SET_TARGET_PROPERTIES(eigen33correctness PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(eigen33correctness
	vcl_core
	vcl_math
)
