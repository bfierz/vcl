project(vcl_graphics_vulkan_test)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Generate test executable
add_executable(vcl_graphics_vulkan_test "")

set(VCL_TEST_SRC
	context.cpp
	shaderprogram.cpp
)
set(VCL_TEST_INC
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
	"vulkan"
	"QuadSpirvVS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_0
)
vclcompileglsl(
	${PROJECT_SOURCE_DIR}/quad.frag
	"vulkan"
	"QuadSpirvFS"
	"${CURR_INC_DIRS}"
	COMPILEDSHADERS_1
)
set(COMPILEDSHADERS ${COMPILEDSHADERS_0} ${COMPILEDSHADERS_1})

target_sources(vcl_graphics_vulkan_test
	PRIVATE
		${VCL_TEST_INC}
		${VCL_TEST_SRC}
		${COMPILEDSHADERS}
)

source_group("" FILES ${VCL_TEST_SRC} ${VCL_TEST_INC})
set_target_properties(vcl_graphics_vulkan_test PROPERTIES FOLDER tests)

target_link_libraries(vcl_graphics_vulkan_test
	vcl_graphics
	gtest
	gtest_main
)