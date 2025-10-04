#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2017 Basil Fierz
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
project(vcl_core_benchmarks)

include(../../cmake/CPM.cmake)

# Status message
message(STATUS "Configuring 'vcl_core_benchmarks'")

CPMAddPackage(
	GITHUB_REPOSITORY foonathan/memory
	VERSION 0.7-4
	OPTIONS
		"FOONATHAN_MEMORY_BUILD_EXAMPLES NO"
		"FOONATHAN_MEMORY_BUILD_TESTS NO"
	EXCLUDE_FROM_ALL
)
set_property(TARGET foonathan_memory PROPERTY FOLDER 3rd-party)
set_property(TARGET foonathan_memory_node_size_debugger PROPERTY FOLDER 3rd-party)

set(SRC
	main.cpp
)
source_group("" FILES ${SRC})

add_executable(vcl_core_benchmarks ${SRC})
set_target_properties(vcl_core_benchmarks PROPERTIES FOLDER benchmarks)

target_link_libraries(vcl_core_benchmarks
	vcl_core
	benchmark
	foonathan_memory
)

if (WIN32)
	target_link_libraries(vcl_core_benchmarks
		windowsapp.lib
	)
endif()
