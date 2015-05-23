PROJECT(vcl_core)

# Write the configuration file
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/vcl/config/config.h.in" "${PROJECT_BINARY_DIR}/vcl/config/config.h")

# Summarize configuration
MESSAGE(STATUS "Writing configuration file")

# VCL / CONFIG
SET(VCL_CONFIG_INC
	${PROJECT_BINARY_DIR}/vcl/config/config.h
	
	vcl/config/compiler.h
	vcl/config/eigen.h
	vcl/config/global.h
	vcl/config/macros.h
)

# VCL / CORE
SET(VCL_CORE_SRC
	vcl/core/contract.cpp
	vcl/core/handle.cpp
)
SET(VCL_CORE_INC
	vcl/core/contract.h
	vcl/core/handle.h
	vcl/core/interleavedarray.h
)

# VCL / CORE / 3RDPARTY
SET(VCL_CORE_3RDPARTY_SRC
	vcl/core/3rdparty/format.cc
)
SET(VCL_CORE_3RDPARTY_INL
)
SET(VCL_CORE_3RDPARTY_INC
	vcl/core/3rdparty/format.h
)

# VCL / CORE / CONTAINER
SET(VCL_CORE_CONTAINER_SRC
)
SET(VCL_CORE_CONTAINER_INL
)
SET(VCL_CORE_CONTAINER_INC
	vcl/core/container/bitvector.h
)

# VCL / CORE / MEMORY
SET(VCL_CORE_MEMORY_SRC
)
SET(VCL_CORE_MEMORY_INL
)
SET(VCL_CORE_MEMORY_INC
	vcl/core/memory/allocator.h
	vcl/core/memory/smart_ptr.h
)

# VCL / CORE / SIMD
SET(VCL_CORE_SIMD_SRC
	vcl/core/simd/intrinsics_avx.cpp
	vcl/core/simd/intrinsics_sse.cpp
)
SET(VCL_CORE_SIMD_INL
)
SET(VCL_CORE_SIMD_INC
	vcl/core/simd/avx_mathfun.h
	vcl/core/simd/sse_mathfun.h
	vcl/core/simd/bool4_sse.h
	vcl/core/simd/bool8_avx.h
	vcl/core/simd/bool8_sse.h
	vcl/core/simd/bool16_avx.h
	vcl/core/simd/bool16_sse.h
	vcl/core/simd/float4_sse.h
	vcl/core/simd/float8_avx.h
	vcl/core/simd/float8_sse.h
	vcl/core/simd/float16_avx.h
	vcl/core/simd/float16_sse.h
	vcl/core/simd/int4_sse.h
	vcl/core/simd/int8_avx.h
	vcl/core/simd/int8_sse.h
	vcl/core/simd/int16_avx.h
	vcl/core/simd/int16_sse.h
	vcl/core/simd/intrinsics_avx.h
	vcl/core/simd/intrinsics_sse.h
	vcl/core/simd/memory.h
	vcl/core/simd/memory_avx.h
	vcl/core/simd/memory_sse.h
	vcl/core/simd/vectorscalar.h
)

# VCL / MATH
SET(VCL_MATH_INC
	vcl/math/math.h
)

# VCL / UTIL
SET(VCL_UTIL_SRC
	vcl/util/precisetimer.cpp
)
SET(VCL_UTIL_INC
	vcl/util/precisetimer.h
)

SOURCE_GROUP(config FILES ${VCL_CONFIG_INC})
SOURCE_GROUP(core\\3rdParty FILES ${VCL_CORE_3RDPARTY_SRC} ${VCL_CORE_3RDPARTY_INL} ${VCL_CORE_3RDPARTY_INC})
SOURCE_GROUP(core\\container FILES ${VCL_CORE_CONTAINER_SRC} ${VCL_CORE_CONTAINER_INL} ${VCL_CORE_CONTAINER_INC})
SOURCE_GROUP(core\\memory FILES ${VCL_CORE_MEMORY_SRC} ${VCL_CORE_MEMORY_INL} ${VCL_CORE_MEMORY_INC})
SOURCE_GROUP(core\\simd FILES ${VCL_CORE_SIMD_SRC} ${VCL_CORE_SIMD_INL} ${VCL_CORE_SIMD_INC})
SOURCE_GROUP(core FILES ${VCL_CORE_SRC} ${VCL_CORE_INC})
SOURCE_GROUP(math FILES ${VCL_MATH_INC})
SOURCE_GROUP(util FILES ${VCL_UTIL_SRC} ${VCL_UTIL_INC})

SET(SOURCE
	${VCL_CONFIG_INC}
	${VCL_CORE_3RDPARTY_SRC} ${VCL_CORE_3RDPARTY_INL} ${VCL_CORE_3RDPARTY_INC}
	${VCL_CORE_CONTAINER_SRC} ${VCL_CORE_CONTAINER_INL} ${VCL_CORE_CONTAINER_INC}
	${VCL_CORE_MEMORY_SRC} ${VCL_CORE_MEMORY_INL} ${VCL_CORE_MEMORY_INC}
	${VCL_CORE_SIMD_SRC} ${VCL_CORE_SIMD_INL} ${VCL_CORE_SIMD_INC}
	${VCL_CORE_SRC} ${VCL_CORE_INC}
	${VCL_MATH_INC}
	${VCL_UTIL_SRC} ${VCL_UTIL_INC}
)

# Generate library
ADD_LIBRARY(vcl_core STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_core PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Now make sure that we find the config.h file
TARGET_INCLUDE_DIRECTORIES(vcl_core PUBLIC ${PROJECT_BINARY_DIR})
