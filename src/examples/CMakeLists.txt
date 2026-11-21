
# Configure the VCL libraries
include(../cmake/VCLCudaSupport.cmake)
include(../cmake/VCLOpenCLSupport.cmake)

add_subdirectory(rtti)
add_subdirectory(simd)

# GLFW is the window system of the OpenGL and Vulkan examples
if((VCL_OPENGL_SUPPORT OR VCL_VULKAN_SUPPORT) AND NOT EMSCRIPTEN)
	find_package(glfw3 CONFIG REQUIRED)
endif()

# The Dear ImGui backends and extensions, compiled once for the examples using them
if(VCL_OPENGL_SUPPORT OR VCL_VULKAN_SUPPORT OR (VCL_D3D12_SUPPORT AND WIN32))
	add_subdirectory(imgui)
endif()

# The D3D12 examples are written against Win32 and DXGI
if(VCL_D3D12_SUPPORT AND WIN32)
	add_subdirectory(d3d12)
endif()


if(VCL_VULKAN_SUPPORT)
	add_subdirectory(vulkan)
endif()

# The WebGPU examples (webgpu/) are excluded until the backend is ported to the current Dawn API


if(VCL_OPENGL_SUPPORT)
	#add_subdirectory(meshviewer)
	add_subdirectory(opengl)
endif()

if(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	add_subdirectory(opencl)
endif()
