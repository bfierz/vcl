#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2016 Basil Fierz
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
project(vcl_geometry_benchmarks)

# Status message
message(STATUS "Configuring 'vcl_geometry_benchmarks'")

FetchContent_Declare(
	gte_library
	GIT_REPOSITORY https://github.com/davideberly/GeometricTools.git
	GIT_TAG        GTE-version-5.9
)
FetchContent_GetProperties(gte_library)
if(NOT gte_library_POPULATED)
	FetchContent_Populate(gte_library)
	message(STATUS "Downloaded Geometric Tools Engine to ${gte_library_SOURCE_DIR}")

	add_library(gte INTERFACE)
	target_sources(gte
		INTERFACE
			${gte_library_SOURCE_DIR}/GTE/Mathematics/DistPointTriangle.h
			${gte_library_SOURCE_DIR}/GTE/Mathematics/DistTriangle3Triangle3.h
			${gte_library_SOURCE_DIR}/GTE/Mathematics/IntrRay3AlignedBox3.h
	)
	target_include_directories(gte INTERFACE "${gte_library_SOURCE_DIR}/GTE")
endif()

set(SRC
	main.cpp
)
source_group("" FILES ${SRC})

add_executable(vcl_geometry_benchmarks ${SRC})
set_target_properties(vcl_geometry_benchmarks PROPERTIES FOLDER benchmarks)

target_link_libraries(vcl_geometry_benchmarks
	vcl_core
	vcl.geometry
	benchmark
	gte
)
