#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2014 Basil Fierz
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
project(vcl.core.test)

include(../../cmake/VCLTest.cmake)

# Check dependency requirements
vcl_check_target(nlohmann_json)

set(SOURCE_FILES
	allocator.cpp
	bitvector.cpp
	convert.cpp
	eigen_simd.cpp
	flags.cpp
	fnv1a.cpp
	gather.cpp
	interleave.cpp
	interleavedarray.cpp
	load.cpp
	math.cpp
	minmax.cpp
	rtti.cpp
	scatter.cpp
	scopeguard.cpp
	simd.cpp
	smart_ptr.cpp
	store.cpp
	waveletnoise.cpp
)
vcl_add_test(vcl.core.test)
vcl_target_sources(vcl.core.test "" ${SOURCE_FILES})
if(${CMAKE_VERSION} VERSION_LESS "3.16.0")
	target_sources(vcl.core.test PRIVATE precompiled.h precompiled.cpp)
	add_precompiled_header(vcl.core.test precompiled.h precompiled.cpp)
else()
	target_precompile_headers(vcl.core.test PRIVATE precompiled.h)
endif()
target_link_libraries(vcl.core.test
	vcl_core
	nlohmann_json
)
