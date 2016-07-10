PROJECT(vcl_graphics_opengl_test)

FILE(GLOB VCL_TEST_SRC
	*.cpp
)

FILE(GLOB VCL_TEST_INC
	*.h
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
	gtest
)