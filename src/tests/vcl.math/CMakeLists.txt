PROJECT(vcl_math_test)

SET(VCL_TEST_SRC
	eigen33.cpp
	poldecomp33.cpp
	qr33.cpp
	svd33.cpp
)
SET(VCL_TEST_INC
)

SET(SOURCE
	${VCL_TEST_SRC} ${VCL_TEST_INC}
)

SOURCE_GROUP("" FILES ${VCL_TEST_SRC} ${VCL_TEST_INC})

# Generate library
ADD_EXECUTABLE(vcl_math_test
	${SOURCE}
)

SET_TARGET_PROPERTIES(vcl_math_test PROPERTIES FOLDER tests)

TARGET_LINK_LIBRARIES(vcl_math_test
	vcl_math
	debug ${VCL_GTEST_LIBRARY_DEBUG}/gtest.lib
	debug ${VCL_GTEST_LIBRARY_DEBUG}/gtest_main.lib
	optimized ${VCL_GTEST_LIBRARY_RELEASE}/gtest.lib
	optimized ${VCL_GTEST_LIBRARY_RELEASE}/gtest_main.lib
)