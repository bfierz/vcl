PROJECT(vcl_core_benchmarks)

# Status message
MESSAGE(STATUS "Configuring 'vcl_core_benchmarks'")

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

ADD_EXECUTABLE(vcl_core_benchmarks ${SRC})
SET_TARGET_PROPERTIES(vcl_core_benchmarks PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(vcl_core_benchmarks
	vcl_core
	benchmark
	foonathan_memory
	Shlwapi
)
