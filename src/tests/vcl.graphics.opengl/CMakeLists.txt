PROJECT(vcl_graphics_opengl_test)

SET(VCL_TEST_SRC
	shaderprogram.cpp
	main.cpp
)
SET(VCL_TEST_INC
)

SET(SOURCE
	${VCL_TEST_SRC} ${VCL_TEST_INC}
)

SOURCE_GROUP("" FILES ${VCL_TEST_SRC} ${VCL_TEST_INC})

# Generate library
ADD_EXECUTABLE(vcl_graphics_opengl_test
	${SOURCE}
)

SET_TARGET_PROPERTIES(vcl_graphics_opengl_test PROPERTIES FOLDER tests)

TARGET_LINK_LIBRARIES(vcl_graphics_opengl_test
	vcl_graphics
	${GLEW_LIBRARY}
	${OPENGL_gl_LIBRARY}
	debug ${VCL_GTEST_LIBRARY_DEBUG}/gtest.lib
	optimized ${VCL_GTEST_LIBRARY_RELEASE}/gtest.lib
)