PROJECT(svd33performance)

# Status message
MESSAGE(STATUS "Configuring 'svd33performance'")

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

ADD_EXECUTABLE(svd33performance ${SRC})
SET_TARGET_PROPERTIES(svd33performance PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(svd33performance
	vcl_core
	vcl_math
)
