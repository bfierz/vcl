PROJECT(vcl_core)

# Find dependencies
FIND_PACKAGE(Eigen3 3.3 REQUIRED)

# Write the configuration file
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/vcl/config/config.h.in" "${PROJECT_BINARY_DIR}/vcl/config/config.h")

# Summarize configuration
MESSAGE(STATUS "Writing configuration file")

# Generate library
ADD_LIBRARY(vcl_core STATIC "")

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
	vcl/core/any.h
	vcl/core/contract.h
	vcl/core/convert.h
	vcl/core/flags.h
	vcl/core/handle.h
	vcl/core/interleavedarray.h
	vcl/core/preprocessor.h
)

# VCL / CORE / CONTAINER
SET(VCL_CORE_CONTAINER_SRC
)
SET(VCL_CORE_CONTAINER_INL
)
SET(VCL_CORE_CONTAINER_INC
	vcl/core/container/array.h
	vcl/core/container/bitvector.h
	vcl/core/container/bucketadapter.h
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
	vcl/core/simd/intrinsics_neon.cpp
)
SET(VCL_CORE_SIMD_INL
)
SET(VCL_CORE_SIMD_INC
	vcl/core/simd/avx_mathfun.h
	vcl/core/simd/sse_mathfun.h
	vcl/core/simd/neon_mathfun.h
	vcl/core/simd/bool4_sse.h
	vcl/core/simd/bool4_neon.h
	vcl/core/simd/bool8_avx.h
	vcl/core/simd/bool8_sse.h
	vcl/core/simd/bool8_neon.h
	vcl/core/simd/bool16_avx.h
	vcl/core/simd/bool16_sse.h
	vcl/core/simd/bool16_neon.h
	vcl/core/simd/float4_sse.h
	vcl/core/simd/float4_neon.h
	vcl/core/simd/float8_avx.h
	vcl/core/simd/float8_sse.h
	vcl/core/simd/float8_neon.h
	vcl/core/simd/float16_avx.h
	vcl/core/simd/float16_sse.h
	vcl/core/simd/float16_neon.h
	vcl/core/simd/int4_sse.h
	vcl/core/simd/int4_neon.h
	vcl/core/simd/int8_avx.h
	vcl/core/simd/int8_sse.h
	vcl/core/simd/int8_neon.h
	vcl/core/simd/int16_avx.h
	vcl/core/simd/int16_sse.h
	vcl/core/simd/int16_neon.h
	vcl/core/simd/intrinsics_avx.h
	vcl/core/simd/intrinsics_sse.h
	vcl/core/simd/intrinsics_neon.h
	vcl/core/simd/memory.h
	vcl/core/simd/memory_avx.h
	vcl/core/simd/memory_sse.h
	vcl/core/simd/memory_neon.h
	vcl/core/simd/vectorscalar.h
)

# VCL / DEBUG / MSVC
SET(VCL_DEBUG_MSVC_INC
	vcl/debug/msvc/qt5.natvis
)

# VCL / MATH
SET(VCL_MATH_INC
	vcl/math/ceil.h
	vcl/math/math.h
)

# VCL / RTTI
SET(VCL_RTTI_SRC
	vcl/rtti/metatypebase.cpp
	vcl/rtti/metatyperegistry.cpp
)
SET(VCL_RTTI_INL
	vcl/rtti/metatypeconstructor.inl
)
SET(VCL_RTTI_INC
	vcl/rtti/attributebase.h
	vcl/rtti/attribute.h
	vcl/rtti/constructorbase.h
	vcl/rtti/constructor.h
	vcl/rtti/factory.h
	vcl/rtti/metatypebase.h
	vcl/rtti/metatypeconstructor.h
	vcl/rtti/metatypelookup.h
	vcl/rtti/metatyperegistry.h
	vcl/rtti/metatype.h
	vcl/rtti/serializer.h
)

# VCL / UTIL
SET(VCL_UTIL_SRC
	vcl/util/precisetimer.cpp
	vcl/util/stringparser.cpp
	vcl/util/vectornoise.cpp
	vcl/util/waveletnoise.cpp
)
SET(VCL_UTIL_INC
	vcl/util/donotoptimizeaway.h
	vcl/util/hashedstring.h
	vcl/util/precisetimer.h
	vcl/util/mortoncodes.h
	vcl/util/reservememory.h
	vcl/util/scopeguard.h
	vcl/util/stringparser.h
	vcl/util/vectornoise.h
	vcl/util/waveletnoise.h
)

SOURCE_GROUP(config FILES ${VCL_CONFIG_INC})
SOURCE_GROUP(core\\container FILES ${VCL_CORE_CONTAINER_SRC} ${VCL_CORE_CONTAINER_INL} ${VCL_CORE_CONTAINER_INC})
SOURCE_GROUP(core\\memory FILES ${VCL_CORE_MEMORY_SRC} ${VCL_CORE_MEMORY_INL} ${VCL_CORE_MEMORY_INC})
SOURCE_GROUP(core\\simd FILES ${VCL_CORE_SIMD_SRC} ${VCL_CORE_SIMD_INL} ${VCL_CORE_SIMD_INC})
SOURCE_GROUP(core FILES ${VCL_CORE_SRC} ${VCL_CORE_INC})
SOURCE_GROUP(debug\\msvc FILES ${VCL_DEBUG_MSVC_INC})
SOURCE_GROUP(math FILES ${VCL_MATH_INC})
SOURCE_GROUP(rtti FILES ${VCL_RTTI_SRC} ${VCL_RTTI_INL} ${VCL_RTTI_INC})
SOURCE_GROUP(util FILES ${VCL_UTIL_SRC} ${VCL_UTIL_INC})

SET(SOURCE
	${VCL_CONFIG_INC}
	${VCL_CORE_CONTAINER_SRC} ${VCL_CORE_CONTAINER_INL} ${VCL_CORE_CONTAINER_INC}
	${VCL_DEBUG_MSVC_INC}
	${VCL_CORE_MEMORY_SRC} ${VCL_CORE_MEMORY_INL} ${VCL_CORE_MEMORY_INC}
	${VCL_CORE_SIMD_SRC} ${VCL_CORE_SIMD_INL} ${VCL_CORE_SIMD_INC}
	${VCL_CORE_SRC} ${VCL_CORE_INC}
	${VCL_MATH_INC}
	${VCL_RTTI_SRC} ${VCL_RTTI_INL} ${VCL_RTTI_INC}
	${VCL_UTIL_SRC} ${VCL_UTIL_INC}
)

TARGET_SOURCES(vcl_core PRIVATE ${SOURCE})
SET_TARGET_PROPERTIES(vcl_core PROPERTIES FOLDER libs)
if(WIN32)
	SET_TARGET_PROPERTIES(vcl_core PROPERTIES DEBUG_POSTFIX _d)
endif()

TARGET_INCLUDE_DIRECTORIES(vcl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Now make sure that the config.h file is found during development
TARGET_INCLUDE_DIRECTORIES(vcl_core PUBLIC
    $<BUILD_INTERFACE:
        ${PROJECT_BINARY_DIR}
    >
)

FUNCTION( prepend_current_dir SOURCE_FILES )
  FOREACH( SOURCE_FILE ${${SOURCE_FILES}} )
    SET( MODIFIED ${MODIFIED} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} )
  ENDFOREACH()
  SET( ${SOURCE_FILES} ${MODIFIED} PARENT_SCOPE )
ENDFUNCTION()

# Export all the natvis files so they will automatically be loaded into Visual Studio
prepend_current_dir(VCL_DEBUG_MSVC_INC)
target_sources(vcl_core INTERFACE 
	${VCL_DEBUG_MSVC_INC}
)

TARGET_LINK_LIBRARIES(vcl_core
	Eigen3
	fmt-header-only
	GSL
	nlohmann_json
)

# Enable static code analysis
OPTION(VCL_ENABLE_CORE_GUIDELINE_CHECKER "Enable core guideline checking")
IF (VCL_ENABLE_CORE_GUIDELINE_CHECKER)
	ENABLE_VS_GUIDELINE_CHECKER(vcl_core)
	TARGET_COMPILE_DEFINITIONS(vcl_core PRIVATE VCL_CHECK_CORE_GUIDELINES)
ENDIF()
OPTION(VCL_ENABLE_CLANG_TIDY "Enable clang-tidy")
IF (VCL_ENABLE_CLANG_TIDY)
	ENABLE_CLANG_TIDY(vcl_core)
ENDIF()

# Setup installation
INSTALL(FILES ${VCL_CONFIG_INC} DESTINATION include/vcl/config)
INSTALL(FILES ${VCL_CORE_CONTAINER_INL} ${VCL_CORE_CONTAINER_INC} DESTINATION include/vcl/core/container)
INSTALL(FILES ${VCL_CORE_MEMORY_INL} ${VCL_CORE_MEMORY_INC} DESTINATION include/vcl/core/memory)
INSTALL(FILES ${VCL_CORE_SIMD_INL} ${VCL_CORE_SIMD_INC} DESTINATION include/vcl/core/simd)
INSTALL(FILES ${VCL_CORE_INC} DESTINATION include/vcl/core)
INSTALL(FILES ${VCL_MATH_INC} DESTINATION include/vcl/math)
INSTALL(FILES ${VCL_RTTI_INL} ${VCL_RTTI_INC} DESTINATION include/vcl/rtti)
INSTALL(FILES ${VCL_UTIL_INC} DESTINATION include/vcl/util)
INSTALL(TARGETS vcl_core ARCHIVE DESTINATION lib)
