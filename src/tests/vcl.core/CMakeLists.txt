PROJECT(vcl_core_test)

SET(VCL_TEST_SRC
	allocator.cpp
	gather.cpp
	interleavedarray.cpp
	load.cpp
	scatter.cpp
)
SET(VCL_TEST_INC
)

SET(SOURCE
	${VCL_TEST_SRC} ${VCL_TEST_INC}
)

SOURCE_GROUP("" FILES ${VCL_TEST_SRC} ${VCL_TEST_INC})

# Generate library
ADD_EXECUTABLE(vcl_core_test
	${SOURCE}
)

SET_TARGET_PROPERTIES(vcl_core_test PROPERTIES FOLDER tests)

TARGET_LINK_LIBRARIES(vcl_core_test
	vcl_core
	debug ${VCL_GTEST_LIBRARY_DEBUG}/gtest.lib
	debug ${VCL_GTEST_LIBRARY_DEBUG}/gtest_main.lib
	optimized ${VCL_GTEST_LIBRARY_RELEASE}/gtest.lib
	optimized ${VCL_GTEST_LIBRARY_RELEASE}/gtest_main.lib
)