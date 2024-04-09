project(vkclearscreen)

set(SRC
	main.cpp
)
source_group("" FILES ${SRC})

add_executable(vkclearscreen ${SRC})
set_target_properties(vkclearscreen PROPERTIES FOLDER examples/vulkan)
target_link_libraries(vkclearscreen
	vcl_graphics
	glfw
)
