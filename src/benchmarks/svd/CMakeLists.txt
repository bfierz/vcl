PROJECT(svdperformance)

# Status message
MESSAGE(STATUS "Configuring 'svdperformance'")

# Find the QtCore library
FIND_PACKAGE(Qt5Core REQUIRED)

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

ADD_EXECUTABLE(svdperformance ${SRC})
SET_TARGET_PROPERTIES(svdperformance PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(svdperformance
	vcl_core
	vcl_math
)
