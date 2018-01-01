PROJECT(vcl_core_test)

# Check dependency requirements
check_target(nlohmann_json)
check_target(gtest)
check_target(gtest_main)

SET(VCL_TEST_SRC
	allocator.cpp
	bitvector.cpp
	flags.cpp
	gather.cpp
	interleavedarray.cpp
	load.cpp
	minmax.cpp
	rtti.cpp
	scatter.cpp
	scopeguard.cpp
	simd.cpp
	smart_ptr.cpp
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
	nlohmann_json
	gtest
	gtest_main
)

# Enable static code analysis
OPTION(VCL_ENABLE_CORE_GUIDELINE_CHECKER "Enable core guideline checking")
IF (VCL_ENABLE_CORE_GUIDELINE_CHECKER)
	ENABLE_VS_GUIDELINE_CHECKER(vcl_core_test)
	TARGET_COMPILE_DEFINITIONS(vcl_core_test PRIVATE VCL_CHECK_CORE_GUIDELINES)
ENDIF()
