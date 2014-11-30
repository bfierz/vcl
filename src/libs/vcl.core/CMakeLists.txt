PROJECT(vcl_core)

# VCL / CONFIG
SET(VCL_CONFIG_INC
	vcl/config/compiler.h
	vcl/config/eigen.h
	vcl/config/global.h
	vcl/config/macros.h
)

# VCL / CORE
SET(VCL_CORE_SRC
)
SET(VCL_CORE_INC
	vcl/core/contract.h
	vcl/core/interleavedarray.h
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
	vcl/core/simd/floatn.h
	vcl/core/simd/intrinsics_avx.h
	vcl/core/simd/intrinsics_sse.h
	vcl/core/simd/memory.h
	vcl/core/simd/memory_avx.h
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
SOURCE_GROUP(core\\simd FILES ${VCL_CORE_SIMD_SRC} ${VCL_CORE_SIMD_INL} ${VCL_CORE_SIMD_INC})
SOURCE_GROUP(core FILES ${VCL_CORE_SRC} ${VCL_CORE_INC})
SOURCE_GROUP(math FILES ${VCL_MATH_INC})
SOURCE_GROUP(util FILES ${VCL_UTIL_SRC} ${VCL_UTIL_INC})

SET(SOURCE
	${VCL_CONFIG_INC}
	${VCL_CORE_SIMD_SRC} ${VCL_CORE_SIMD_INL} ${VCL_CORE_SIMD_INC}
	${VCL_CORE_SRC} ${VCL_CORE_INC}
	${VCL_MATH_INC}
	${VCL_UTIL_SRC} ${VCL_UTIL_INC}
)

# Generate library
ADD_LIBRARY(vcl_core STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_core PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
