project(vcl_graphics_opengl_test)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# GLSL compiler
INCLUDE(${CMAKE_SOURCE_DIR}/cmake/VCLCompileGLSL.cmake)

# Generate test executable
add_executable(vcl_graphics_opengl_test "")

set(VCL_TEST_SRC
	blendstate.cpp
	buffers.cpp
	depthstencilstate.cpp
	engine.cpp
	histogram.cpp
	imageprocessor.cpp
	main.cpp
	radixsort.cpp
	rasterstate.cpp
	scan.cpp
	shaderprogram.cpp
	textures.cpp
)

set(VCL_TEST_INC
	bitmap.h
	pattern.h
)

set(SHADERS
	quad.vert
	quad.frag
)

# Access the include directories in order to compile the GLSL shaders
get_property(CURR_INC_DIRS_0 TARGET vcl_graphics PROPERTY INCLUDE_DIRECTORIES)
list(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CURR_INC_DIRS_0})

vclcompileglsl(
	${PROJECT_SOURCE_DIR}/quad.vert
	"QuadSpirvVS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_0
)
vclcompileglsl(
	${PROJECT_SOURCE_DIR}/quad.frag
	"QuadSpirvFS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_1
)
set(COMPILEDSHADERS ${COMPILEDSHADERS_0} ${COMPILEDSHADERS_1})

target_sources(vcl_graphics_opengl_test
	PRIVATE
		${VCL_TEST_INC}
		${VCL_TEST_SRC}
		${COMPILEDSHADERS}
)

target_link_libraries(vcl_graphics_opengl_test
	vcl_graphics
	gtest
)

# Visual Studio configuration
source_group("" FILES ${VCL_TEST_SRC} ${VCL_TEST_INC} ${SHADERS})
set_target_properties(vcl_graphics_opengl_test PROPERTIES FOLDER tests)
