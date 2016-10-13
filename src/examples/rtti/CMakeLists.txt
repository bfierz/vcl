PROJECT(simple_rtti)

SET(SIMPLE_SRC
	baseobject.cpp
	baseobject.h
	simple.cpp
)

SOURCE_GROUP("" FILES ${SIMPLE_SRC})

ADD_EXECUTABLE(simple_rtti ${SIMPLE_SRC})
SET_TARGET_PROPERTIES(simple_rtti PROPERTIES FOLDER examples/rtti)
TARGET_LINK_LIBRARIES(simple_rtti
	vcl_core
)
