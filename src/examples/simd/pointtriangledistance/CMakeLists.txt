PROJECT(pointtriangledistance)

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

ADD_EXECUTABLE(pointtriangledistance ${SRC})
SET_TARGET_PROPERTIES(pointtriangledistance PROPERTIES FOLDER examples/simd)

TARGET_LINK_LIBRARIES(pointtriangledistance
	vcl_core
	vcl_geometry
)
