project(vcl_core)

# Configure the VCL compilation environment
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/VCLConfig.cmake)

# Check dependency requirements
vcl_check_target(${vcl_ext_absl})
vcl_check_target(${vcl_ext_eigen})
vcl_check_target(${vcl_ext_fmt})
vcl_check_target(${vcl_ext_gsl})

# Generate library
add_library(vcl_core STATIC "")
vcl_configure(vcl_core)

# Write the configuration file
configure_file("${PROJECT_SOURCE_DIR}/vcl/config/config.h.in" "${PROJECT_BINARY_DIR}/vcl/config/config.h")

# Summarize configuration
message(STATUS "Writing configuration file")

# VCL / CONFIG
set(VCL_CONFIG_INC
	${PROJECT_BINARY_DIR}/vcl/config/config.h
	
	vcl/config/compiler.h
	vcl/config/eigen.h
	vcl/config/global.h
	vcl/config/macros.h
)

# VCL / CORE
set(VCL_CORE_SRC
	vcl/core/contract.cpp
	vcl/core/handle.cpp
)
set(VCL_CORE_INC
	vcl/core/any.h
	vcl/core/contract.h
	vcl/core/convert.h
	vcl/core/flags.h
	vcl/core/handle.h
	vcl/core/interleavedarray.h
	vcl/core/preprocessor.h
)

# VCL / CORE / CONTAINER
set(VCL_CORE_CONTAINER_SRC
)
set(VCL_CORE_CONTAINER_INL
)
set(VCL_CORE_CONTAINER_INC
	vcl/core/container/array.h
	vcl/core/container/bitvector.h
	vcl/core/container/bucketadapter.h
)

# VCL / CORE / MEMORY
set(VCL_CORE_MEMORY_SRC
)
set(VCL_CORE_MEMORY_INL
)
set(VCL_CORE_MEMORY_INC
	vcl/core/memory/allocator.h
	vcl/core/memory/smart_ptr.h
)

# VCL / CORE / SIMD
set(VCL_CORE_SIMD_SRC
	vcl/core/simd/intrinsics_avx.cpp
	vcl/core/simd/intrinsics_sse.cpp
	vcl/core/simd/intrinsics_neon.cpp
)
set(VCL_CORE_SIMD_INL
)
set(VCL_CORE_SIMD_INC
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
set(VCL_DEBUG_MSVC_INC
	vcl/debug/msvc/eigen.natvis
	vcl/debug/msvc/qt5.natvis
)

# VCL / MATH
set(VCL_MATH_INC
	vcl/math/ceil.h
	vcl/math/math.h
)

# VCL / RTTI
set(VCL_RTTI_SRC
	vcl/rtti/metatypebase.cpp
	vcl/rtti/metatyperegistry.cpp
)
set(VCL_RTTI_INL
	vcl/rtti/metatypeconstructor.inl
)
set(VCL_RTTI_INC
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
set(VCL_UTIL_SRC
	vcl/util/precisetimer.cpp
	vcl/util/stringparser.cpp
	vcl/util/vectornoise.cpp
	vcl/util/waveletnoise.cpp
)
set(VCL_UTIL_INC
	vcl/util/donotoptimizeaway.h
	vcl/util/hashedstring.h
	vcl/util/precisetimer.h
	vcl/util/mortoncodes.h
	vcl/util/reservememory.h
	vcl/util/scopeguard.h
	vcl/util/stringparser.h
	vcl/util/vectornoise.h
	vcl/util/waveletnoise.h
	vcl/util/waveletnoise_modulo.h
)

source_group(config FILES ${VCL_CONFIG_INC})
source_group(core\\container FILES ${VCL_CORE_CONTAINER_SRC} ${VCL_CORE_CONTAINER_INL} ${VCL_CORE_CONTAINER_INC})
source_group(core\\memory FILES ${VCL_CORE_MEMORY_SRC} ${VCL_CORE_MEMORY_INL} ${VCL_CORE_MEMORY_INC})
source_group(core\\simd FILES ${VCL_CORE_SIMD_SRC} ${VCL_CORE_SIMD_INL} ${VCL_CORE_SIMD_INC})
source_group(core FILES ${VCL_CORE_SRC} ${VCL_CORE_INC})
source_group(debug\\msvc FILES ${VCL_DEBUG_MSVC_INC})
source_group(math FILES ${VCL_MATH_INC})
source_group(rtti FILES ${VCL_RTTI_SRC} ${VCL_RTTI_INL} ${VCL_RTTI_INC})
source_group(util FILES ${VCL_UTIL_SRC} ${VCL_UTIL_INC})

set(SOURCE
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

target_sources(vcl_core PRIVATE ${SOURCE})
set_target_properties(vcl_core PROPERTIES FOLDER libs)
if(WIN32)
	set_target_properties(vcl_core PROPERTIES DEBUG_POSTFIX _d)
endif()

target_include_directories(vcl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Now make sure that the config.h file is found during development
target_include_directories(vcl_core PUBLIC
	$<BUILD_INTERFACE:
		${PROJECT_BINARY_DIR}
	>
)

function( prepend_current_dir SOURCE_FILES )
  foreach( SOURCE_FILE ${${SOURCE_FILES}} )
	set( MODIFIED ${MODIFIED} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} )
  endforeach()
  set( ${SOURCE_FILES} ${MODIFIED} PARENT_SCOPE )
endfunction()

# Export all the natvis files so they will automatically be loaded into Visual Studio
prepend_current_dir(VCL_DEBUG_MSVC_INC)
target_sources(vcl_core INTERFACE 
	${VCL_DEBUG_MSVC_INC}
)

target_link_libraries(vcl_core
	${vcl_ext_absl}
	${vcl_ext_eigen}
	${vcl_ext_fmt}
	${vcl_ext_gsl}
)

# Enable static code analysis
if(VCL_ENABLE_CORE_GUIDELINE_CHECKER)
	enable_vs_guideline_checker(vcl_core)
endif()
if(VCL_ENABLE_CLANG_TIDY)
	enable_clang_tidy(vcl_core)
endif()

# Setup installation
set(INCLUDE_DIR_PREFIX "vcl_core/include")
set(LIBRARY_DIR_PREFIX "vcl_core/lib")
install(FILES ${VCL_CONFIG_INC} DESTINATION ${INCLUDE_DIR_PREFIX}/vcl/config)
install(FILES ${VCL_CORE_CONTAINER_INL} ${VCL_CORE_CONTAINER_INC} DESTINATION ${INCLUDE_DIR_PREFIX}/vcl/core/container)
install(FILES ${VCL_CORE_MEMORY_INL} ${VCL_CORE_MEMORY_INC} DESTINATION ${INCLUDE_DIR_PREFIX}/vcl/core/memory)
install(FILES ${VCL_CORE_SIMD_INL} ${VCL_CORE_SIMD_INC} DESTINATION ${INCLUDE_DIR_PREFIX}/vcl/core/simd)
install(FILES ${VCL_CORE_INC} DESTINATION ${INCLUDE_DIR_PREFIX}/vcl/core)
install(FILES ${VCL_MATH_INC} DESTINATION ${INCLUDE_DIR_PREFIX}/vcl/math)
install(FILES ${VCL_RTTI_INL} ${VCL_RTTI_INC} DESTINATION ${INCLUDE_DIR_PREFIX}/vcl/rtti)
install(FILES ${VCL_UTIL_INC} DESTINATION ${INCLUDE_DIR_PREFIX}/vcl/util)
install(TARGETS vcl_core ARCHIVE DESTINATION ${LIBRARY_DIR_PREFIX})
