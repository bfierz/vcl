PROJECT(vcl_math_cuda_test)

SET(VCL_TEST_SRC
	redux.cpp
)
SET(VCL_TEST_INC
)

SET(SOURCE
	${VCL_TEST_SRC} ${VCL_TEST_INC}
)

SOURCE_GROUP("" FILES ${VCL_TEST_SRC} ${VCL_TEST_INC})

# Generate library
ADD_EXECUTABLE(vcl_math_cuda_test
	${SOURCE}
)

SET_TARGET_PROPERTIES(vcl_math_cuda_test PROPERTIES FOLDER tests)

TARGET_LINK_LIBRARIES(vcl_math_cuda_test
	vcl_math_cuda
	gtest
	gtest_main
)