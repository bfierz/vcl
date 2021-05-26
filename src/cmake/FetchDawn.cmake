#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2021 Basil Fierz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
include(FetchContent)
if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
	get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

	if ((IS_MULTI_CONFIG AND "Debug" IN_LIST CMAKE_CONFIGURATION_TYPES) OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
		FetchContent_Declare(
			webgpu_library_debug
			URL      https://github.com/bfierz/dawn-builds/releases/download/0.0.20210521-41c87d97/dawn_win_x64_debug.zip
		)
		FetchContent_GetProperties(webgpu_library_debug)
		if(NOT webgpu_library_debug_POPULATED)
			FetchContent_Populate(webgpu_library_debug)
			message(STATUS "Downloaded Google Dawn Debug to ${webgpu_library_debug_SOURCE_DIR}")
		endif()
	endif()
	if (IS_MULTI_CONFIG OR NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
		FetchContent_Declare(
			webgpu_library_release
			URL      https://github.com/bfierz/dawn-builds/releases/download/0.0.20210521-41c87d97/dawn_win_x64_release.zip
		)
		FetchContent_GetProperties(webgpu_library_release)
		if(NOT webgpu_library_release_POPULATED)
			FetchContent_Populate(webgpu_library_release)
			message(STATUS "Downloaded Google Dawn Release to ${webgpu_library_release_SOURCE_DIR}")
		endif()
	endif()
elseif("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Linux")
	FetchContent_Declare(
		webgpu_library_release
		URL      https://github.com/bfierz/dawn-builds/releases/download/0.0.20210521-41c87d97/dawn_linux_x64_release.zip
	)
	FetchContent_GetProperties(webgpu_library_release)
	if(NOT webgpu_library_release_POPULATED)
		FetchContent_Populate(webgpu_library_release)
		message(STATUS "Downloaded Google Dawn to ${webgpu_library_release_SOURCE_DIR}")
	endif()
endif()

set(DAWN_SORCE_DIRECTORY "$<$<CONFIG:Debug>:${webgpu_library_debug_SOURCE_DIR}>$<$<NOT:$<CONFIG:Debug>>:${webgpu_library_release_SOURCE_DIR}>")
set(DAWN_INCLUDE_DIR "${DAWN_SORCE_DIRECTORY}/include" CACHE PATH "Dawn WebGPU include path")
set(DAWN_NATIVE_LIBRARY "${DAWN_SORCE_DIRECTORY}/lib/dawn_native.dll.lib" CACHE FILEPATH "Dawn WebGPU library")
set(DAWN_PROC_LIBRARY "${DAWN_SORCE_DIRECTORY}/lib/dawn_proc.dll.lib" CACHE FILEPATH "Dawn WebGPU procedure hooks library")

set(WEBGPU_INCLUDE_DIR "${DAWN_SORCE_DIRECTORY}/include" CACHE PATH "WebGPU include path")
set(WEBGPU_CPP_LIBRARY "${DAWN_SORCE_DIRECTORY}/lib/webgpu_cpp.lib" CACHE FILEPATH "WebGPU C++ library")
