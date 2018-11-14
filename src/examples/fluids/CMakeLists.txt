PROJECT(fluids)

SET(INC
	bitmap.h
)

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC} ${INC})

ADD_EXECUTABLE(fluids
	${INC} ${SRC}
)
SET_TARGET_PROPERTIES(fluids PROPERTIES FOLDER examples)

TARGET_LINK_LIBRARIES(fluids
	vcl_physics_fluid
)
