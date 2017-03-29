PROJECT(vcl_math_test)

SET(VCL_TEST_SRC
	eigen33.cpp
	poisson1d.cpp
	poisson2d.cpp
	poisson3d.cpp
	poldecomp33.cpp
	qr33.cpp
	rotation33.cpp
	svd33.cpp
)
SET(VCL_TEST_INC
	poisson.h
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
	gtest
	gtest_main
)