PROJECT(enumcl)

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

ADD_EXECUTABLE(enumcl ${SRC})
SET_TARGET_PROPERTIES(enumcl PROPERTIES FOLDER examples/opencl)

TARGET_LINK_LIBRARIES(enumcl
	vcl_compute_opencl
)
