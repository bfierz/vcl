
# Configure the VCL libraries
include(../cmake/VCLCudaSupport.cmake)
include(../cmake/VCLOpenCLSupport.cmake)

add_subdirectory(rtti)
add_subdirectory(simd)

if((VCL_OPENGL_SUPPORT OR VCL_WEBGPU_SUPPORT) AND NOT EMSCRIPTEN)
	add_subdirectory(3rdparty/glfw EXCLUDE_FROM_ALL)
	set_property(TARGET glfw PROPERTY FOLDER 3rd-party)
endif()

if(VCL_D3D12_SUPPORT)
	add_subdirectory(d3d12)
endif()


if(VCL_VULKAN_SUPPORT)
	add_subdirectory(vulkan)
endif()

if(VCL_WEBGPU_SUPPORT)
	add_subdirectory(webgpu)
endif()


if(VCL_OPENGL_SUPPORT)
	add_subdirectory(meshviewer)
	add_subdirectory(opengl)
endif()

if(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	add_subdirectory(opencl)
endif()
