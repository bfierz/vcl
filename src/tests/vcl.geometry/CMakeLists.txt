PROJECT(vcl_geometry_test)

SET(VCL_TEST_SRC
	distance.cpp
)
SET(VCL_TEST_INC
)

SET(SOURCE
	${VCL_TEST_SRC} ${VCL_TEST_INC}
)

SOURCE_GROUP("" FILES ${VCL_TEST_SRC} ${INCVCL_TEST_INC})

# Generate library
ADD_EXECUTABLE(vcl_geometry_test
	${SOURCE}
)

SET_TARGET_PROPERTIES(vcl_geometry_test PROPERTIES FOLDER tests)

TARGET_LINK_LIBRARIES(vcl_geometry_test
	vcl_geometry
	debug ${VCL_GTEST_LIBRARY_DEBUG}/gtest.lib
	debug ${VCL_GTEST_LIBRARY_DEBUG}/gtest_main.lib
	optimized ${VCL_GTEST_LIBRARY_RELEASE}/gtest.lib
	optimized ${VCL_GTEST_LIBRARY_RELEASE}/gtest_main.lib
)