PROJECT(distancebenchmark)

SET(VCL_GTE_INCLUDE CACHE PATH "Include directory of GeometricToolsEngine")

# Status message
MESSAGE(STATUS "Configuring 'distancebenchmark'")

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

INCLUDE_DIRECTORIES(${VCL_GTE_INCLUDE})
ADD_EXECUTABLE(distancebenchmark ${SRC})
SET_TARGET_PROPERTIES(distancebenchmark PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(distancebenchmark
	vcl_core
	vcl_geometry
	debug ${VCL_GOOGLEBENCHMARK_LIBRARY_DEBUG}
	optimized ${VCL_GOOGLEBENCHMARK_LIBRARY_RELEASE}
	Shlwapi
)
