PROJECT(vcl_geometry_benchmarks)

SET(VCL_GTE_INCLUDE CACHE PATH "Include directory of GeometricToolsEngine")

# Status message
MESSAGE(STATUS "Configuring 'vcl_geometry_benchmarks'")

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

INCLUDE_DIRECTORIES(${VCL_GTE_INCLUDE})
ADD_EXECUTABLE(vcl_geometry_benchmarks ${SRC})
SET_TARGET_PROPERTIES(vcl_geometry_benchmarks PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(vcl_geometry_benchmarks
	vcl_core
	vcl_geometry
	benchmark
	Shlwapi
)
