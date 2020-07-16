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
project(vcl_core)

# Configure the VCL compilation environment
include(../../cmake/VCLLibrary.cmake)

# Check dependency requirements
vcl_check_target(${vcl_ext_absl})
vcl_check_target(${vcl_ext_eigen})
vcl_check_target(${vcl_ext_fmt})

# Generate library
vcl_add_library(vcl_core)

# Write the configuration file
configure_file("${PROJECT_SOURCE_DIR}/vcl/config/config.h.in" "${PROJECT_BINARY_DIR}/vcl/config/config.h")

# Summarize configuration
message(STATUS "Writing configuration file")

set(SOURCE
	vcl/config/compiler.h
	vcl/config/eigen.h
	vcl/config/global.h
	vcl/config/macros.h

	vcl/core/any.h
	vcl/core/contract.cpp
	vcl/core/contract.h
	vcl/core/convert.h
	vcl/core/enum.h
	vcl/core/flags.h
	vcl/core/handle.cpp
	vcl/core/handle.h
	vcl/core/interleavedarray.h
	vcl/core/preprocessor.h
	vcl/core/span.h
	vcl/core/string_view.h

	vcl/core/container/array.h
	vcl/core/container/bitvector.h
	vcl/core/container/bucketadapter.h

	vcl/core/memory/allocator.h
	vcl/core/memory/smart_ptr.h

	vcl/core/simd/common.h
	vcl/core/simd/vectorscalar.h

	vcl/core/simd/bool4_ref.h
	vcl/core/simd/bool8_ref.h
	vcl/core/simd/bool16_ref.h
	vcl/core/simd/float4_ref.h
	vcl/core/simd/float8_ref.h
	vcl/core/simd/float16_ref.h
	vcl/core/simd/int4_ref.h
	vcl/core/simd/int8_ref.h
	vcl/core/simd/int16_ref.h

	vcl/core/simd/bool4_sse.h
	vcl/core/simd/bool8_sse.h
	vcl/core/simd/bool16_sse.h
	vcl/core/simd/float4_sse.h
	vcl/core/simd/float8_sse.h
	vcl/core/simd/float16_sse.h
	vcl/core/simd/int4_sse.h
	vcl/core/simd/int8_sse.h
	vcl/core/simd/int16_sse.h

	vcl/core/simd/bool8_avx.h
	vcl/core/simd/bool16_avx.h
	vcl/core/simd/float8_avx.h
	vcl/core/simd/float16_avx.h
	vcl/core/simd/int4_neon.h
	vcl/core/simd/int8_avx.h
	vcl/core/simd/int8_neon.h
	vcl/core/simd/int16_avx.h
	vcl/core/simd/int16_neon.h

	vcl/core/simd/bool4_neon.h
	vcl/core/simd/bool8_neon.h
	vcl/core/simd/bool16_neon.h
	vcl/core/simd/float4_neon.h
	vcl/core/simd/float8_neon.h
	vcl/core/simd/float16_neon.h
	
	vcl/core/simd/avx_mathfun.h
	vcl/core/simd/sse_mathfun.h
	vcl/core/simd/neon_mathfun.h

	vcl/core/simd/intrinsics_avx.cpp
	vcl/core/simd/intrinsics_avx.h
	vcl/core/simd/intrinsics_sse.cpp
	vcl/core/simd/intrinsics_sse.h
	vcl/core/simd/intrinsics_neon.cpp
	vcl/core/simd/intrinsics_neon.h

	vcl/core/simd/memory.h
	vcl/core/simd/memory_avx.h
	vcl/core/simd/memory_sse.h
	vcl/core/simd/memory_neon.h

	vcl/debug/msvc/abseil.natvis
	vcl/debug/msvc/eigen.natvis
	vcl/debug/msvc/qt5.natvis

	vcl/math/ceil.h
	vcl/math/math.h

	vcl/rtti/attributebase.h
	vcl/rtti/attribute.h
	vcl/rtti/constructorbase.h
	vcl/rtti/constructor.h
	vcl/rtti/factory.h
	vcl/rtti/metatypebase.cpp
	vcl/rtti/metatypebase.h
	vcl/rtti/metatypeconstructor.h
	vcl/rtti/metatypeconstructor.inl
	vcl/rtti/metatypelookup.h
	vcl/rtti/metatyperegistry.cpp
	vcl/rtti/metatyperegistry.h
	vcl/rtti/metatype.h
	vcl/rtti/serializer.h

	vcl/util/donotoptimizeaway.h
	vcl/util/hashedstring.h
	vcl/util/precisetimer.cpp
	vcl/util/precisetimer.h
	vcl/util/mortoncodes.h
	vcl/util/reservememory.cpp
	vcl/util/reservememory.h
	vcl/util/scopeguard.h
	vcl/util/stringparser.cpp
	vcl/util/stringparser.h
	vcl/util/vectornoise.cpp
	vcl/util/vectornoise.h
	vcl/util/waveletnoise.cpp
	vcl/util/waveletnoise.h
	vcl/util/waveletnoise_helpers.h
	vcl/util/waveletnoise_modulo.h
)
vcl_target_sources(vcl_core "vcl" ${SOURCE})

target_sources(vcl_core PRIVATE ${PROJECT_BINARY_DIR}/vcl/config/config.h)
source_group(config FILES ${PROJECT_BINARY_DIR}/vcl/config/config.h)

# Now make sure that the config.h file is found during development
target_include_directories(vcl_core PUBLIC
	$<BUILD_INTERFACE:
		${PROJECT_BINARY_DIR}
	>
)

target_link_libraries(vcl_core
	${vcl_ext_absl}
	${vcl_ext_eigen}
	${vcl_ext_fmt}
)

# Setup installation
set(INCLUDE_DIR_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(LIBRARY_DIR_PREFIX "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY vcl DESTINATION ${INCLUDE_DIR_PREFIX})
install(FILES ${PROJECT_BINARY_DIR}/vcl/config/config.h DESTINATION ${INCLUDE_DIR_PREFIX}/vcl/config)
install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION ${LIBRARY_DIR_PREFIX})
