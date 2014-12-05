PROJECT(meshvertexnormals)

SET(INC
	liver.h
)

SET(SRC
	main.cpp
	liver.cpp
)

SOURCE_GROUP("" FILES ${SRC} ${INC})

ADD_EXECUTABLE(meshvertexnormals
	${INC} ${SRC}
)
SET_TARGET_PROPERTIES(meshvertexnormals PROPERTIES FOLDER examples/simd)

TARGET_LINK_LIBRARIES(meshvertexnormals
	vcl_core
)
