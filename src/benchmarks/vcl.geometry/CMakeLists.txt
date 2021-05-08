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

include(../../cmake/CPM.cmake)

# Status message
message(STATUS "Configuring 'vcl_geometry_benchmarks'")

CPMAddPackage("gh:davideberly/GeometricTools#GTE-version-5.9")
if(GeometricTools_ADDED)
	add_library(gte INTERFACE)
	target_sources(gte
		INTERFACE
			${GeometricTools_SOURCE_DIR}/GTE/Mathematics/DistPointTriangle.h
			${GeometricTools_SOURCE_DIR}/GTE/Mathematics/DistTriangle3Triangle3.h
			${GeometricTools_SOURCE_DIR}/GTE/Mathematics/IntrRay3AlignedBox3.h
	)
	target_include_directories(gte INTERFACE "${GeometricTools_SOURCE_DIR}/GTE")
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
