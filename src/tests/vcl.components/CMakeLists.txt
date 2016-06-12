PROJECT(vcl_components_test)

SET(VCL_TEST_SRC
	entitymanager.cpp
)
SET(VCL_TEST_INC
)

SET(SOURCE
	${VCL_TEST_SRC} ${VCL_TEST_INC}
)

SOURCE_GROUP("" FILES ${VCL_TEST_SRC} ${VCL_TEST_INC})

# Generate library
ADD_EXECUTABLE(vcl_components_test
	${SOURCE}
)

SET_TARGET_PROPERTIES(vcl_components_test PROPERTIES FOLDER tests)

TARGET_LINK_LIBRARIES(vcl_components_test
	vcl_components
	gtest
	gtest_main
)