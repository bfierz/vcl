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

include(${CMAKE_CURRENT_LIST_DIR}/VCLClangTidy.cmake)

# Configure the compiler options for a VCL target
function(vcl_configure tgt)

	message(STATUS "Configure target: ${tgt}")

	# Determine the compiler vendor
	message(STATUS "Detecting compiler: ${CMAKE_CXX_COMPILER_ID}")

	if(${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
		set(VCL_COMPILER_ICC ON)
	elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
		set(VCL_COMPILER_GNU ON)
	elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
		set(VCL_COMPILER_CLANG ON)
	elseif(MSVC)
		set(VCL_COMPILER_MSVC ON)
	else()
		set(VCL_COMPILER_UNKNOWN ON)
	endif()

	# Define C++ standard, minimum requirement is C++14
	# As MSVC is not able to define the minimum level, software needs
	# to implement per feature detection
	set(VCL_CXX_STANDARD "14" CACHE STRING "C++ standard")
	set_property(CACHE VCL_CXX_STANDARD PROPERTY STRINGS "14" "17")
	
	set_target_properties(${tgt} PROPERTIES
		CXX_STANDARD ${VCL_CXX_STANDARD}
		CXX_STANDARD_REQUIRED YES
		CXX_EXTENSIONS NO
	)
	message(STATUS "Using C++${VCL_CXX_STANDARD}")
	
	# Enable clang-tidy for all projects
	if(VCL_COMPILER_CLANG AND VCL_ENABLE_CLANG_TIDY)
		enable_clang_tidy(${tgt})
	endif()
	
	# Determine platform architecture
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(VCL_ADDRESS_SIZE "64")
	else()
		set(VCL_ADDRESS_SIZE "32")
	endif()
	message(STATUS "Compiling for ${VCL_ADDRESS_SIZE}bit machine")

	# Determine the underlying OS
	message(STATUS "Running on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

	# Enable Visual Studio solution folders
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)

	# Control OpenMP support
	option(VCL_OPENMP_SUPPORT "Enable OpenMP support")
	if(VCL_OPENMP_SUPPORT)
		find_package(OpenMP)
		if(OPENMP_FOUND)
			target_compile_options(${tgt} PUBLIC ${OpenMP_CXX_FLAGS})
		endif()
	endif()

	# Define vectorisation
	set(VCL_VECTORIZE "SSE 4.1" CACHE STRING "Vectorization instruction set")
	set_property(CACHE VCL_VECTORIZE PROPERTY STRINGS "SSE 2" "SSE 3" "SSSE 3" "SSE 4.1" "SSE 4.2" "AVX" "AVX 2" "NEON")

	if(VCL_VECTORIZE STREQUAL "SSE 2")
		set(VCL_VECTORIZE_SSE2 TRUE PARENT_SCOPE)
		set(VCL_VECTORIZE_SSE2 TRUE)
	elseif(VCL_VECTORIZE STREQUAL "SSE 3")
		set(VCL_VECTORIZE_SSE3 TRUE PARENT_SCOPE)
		set(VCL_VECTORIZE_SSE3 TRUE)
	elseif(VCL_VECTORIZE STREQUAL "SSSE 3")
		set(VCL_VECTORIZE_SSSE3 TRUE PARENT_SCOPE)
		set(VCL_VECTORIZE_SSSE3 TRUE)
	elseif(VCL_VECTORIZE STREQUAL "SSE 4.1")
		set(VCL_VECTORIZE_SSE4_1 TRUE PARENT_SCOPE)
		set(VCL_VECTORIZE_SSE4_1 TRUE)
	elseif(VCL_VECTORIZE STREQUAL "SSE 4.2")
		set(VCL_VECTORIZE_SSE4_2 TRUE PARENT_SCOPE)
		set(VCL_VECTORIZE_SSE4_2 TRUE)
	elseif(VCL_VECTORIZE STREQUAL "AVX")
		set(VCL_VECTORIZE_AVX TRUE PARENT_SCOPE)
		set(VCL_VECTORIZE_AVX TRUE)
	elseif(VCL_VECTORIZE STREQUAL "AVX 2")
		set(VCL_VECTORIZE_AVX2 TRUE PARENT_SCOPE)
		set(VCL_VECTORIZE_AVX2 TRUE)
	elseif(VCL_VECTORIZE STREQUAL "NEON")
		set(VCL_VECTORIZE_NEON TRUE PARENT_SCOPE)
		set(VCL_VECTORIZE_NEON TRUE)
	endif()
	message(STATUS "Compiling for ${VCL_VECTORIZE}")

	# Set whether contracts should be used
	set(VCL_USE_CONTRACTS CACHE BOOL "Enable contracts")

	# Configure MSVC compiler
	if(VCL_COMPILER_MSVC)
		# Configure release configuration
		target_compile_options(${tgt} PUBLIC "$<$<CONFIG:RELEASE>:/GS->" "$<$<CONFIG:RELEASE>:/fp:fast>")
	
		# Configure all configuration
		# * Enable all warnings
		# * Exceptions
		# * RTTI
		# * Don't be permissive
		target_compile_options(${tgt} PUBLIC "/EHsc" "/GR")
		target_compile_options(${tgt} PRIVATE "/W4" "/permissive-")
	
		# Make AVX available
		if(VCL_VECTORIZE_AVX2)
			target_compile_options(${tgt} PUBLIC "/arch:AVX2")
		elseif(VCL_VECTORIZE_AVX)
			target_compile_options(${tgt} PUBLIC "/arch:AVX")
		elseif(VCL_VECTORIZE_SSE2 AND VCL_ADDRESS_SIZE EQUAL "32")
			# All x64 bit machine come with SSE2, thus it's defined as default
			target_compile_options(${tgt} PUBLIC "/arch:SSE2")
		elseif(VCL_VECTORIZE_NEON)
			target_compile_options(${tgt} PUBLIC "/arch:VFPv4")
		endif()

	endif(VCL_COMPILER_MSVC)

	# Configure GCC and CLANG
	if(VCL_COMPILER_GNU OR VCL_COMPILER_CLANG OR VCL_COMPILER_ICC)

		# Configure all configuration
		# * Enable all warnings
		target_compile_options(${tgt} PUBLIC "-Wall")
		if(VCL_COMPILER_CLANG)
			target_compile_options(${tgt} PUBLIC "-Wno-ignored-attributes" "-D__STRICT_ANSI__" "-Wno-c++98-compat" "-Wno-c++98-compat-pedantic")
		endif()
	
		if(VCL_VECTORIZE_AVX2)
			target_compile_options(${tgt} PUBLIC "-mavx2")
		elseif(VCL_VECTORIZE_AVX)
			target_compile_options(${tgt} PUBLIC "-mavx")
		elseif(VCL_VECTORIZE_SSE4_2)
			target_compile_options(${tgt} PUBLIC "-msse4.2")
		elseif(VCL_VECTORIZE_SSE4_1)
			target_compile_options(${tgt} PUBLIC "-msse4.1")
		elseif(VCL_VECTORIZE_SSSE3)
			target_compile_options(${tgt} PUBLIC "-mssse3")
		elseif(VCL_VECTORIZE_SSE3)
			target_compile_options(${tgt} PUBLIC "-msse3")
		elseif(VCL_VECTORIZE_SSE2)
			target_compile_options(${tgt} PUBLIC "-msse2")
		endif()
	endif(VCL_COMPILER_GNU OR VCL_COMPILER_CLANG OR VCL_COMPILER_ICC)
endfunction()

# Travers all targets
function(get_include_paths OUTPUT_LIST TARGET)
    list(APPEND VISITED_TARGETS "${TARGET}")

	# Determine target type. IMPORTED and INTERFACE_LIBRARY only support
	# a restricted interface.
    get_target_property(IMPORTED "${TARGET}" IMPORTED)
    get_target_property(TYPE "${TARGET}" TYPE)

	# Query the possible links for the recursive search
    if (IMPORTED OR TYPE STREQUAL "INTERFACE_LIBRARY")
        get_target_property(TARGETS "${TARGET}" INTERFACE_LINK_LIBRARIES)
    else()
        get_target_property(TARGETS "${TARGET}" LINK_LIBRARIES)
    endif()

    set(INCLUDE_PATHS "")
    foreach(TGT ${TARGETS})
        if (TARGET ${TGT})
            list(FIND VISITED_TARGETS ${TGT} VISITED)
            if (${VISITED} EQUAL -1)
				get_target_property(TGT_INC_PATHS ${TGT} INTERFACE_INCLUDE_DIRECTORIES)
				foreach(PATH ${TGT_INC_PATHS})
					string(FIND ${PATH} ${PROJECT_SOURCE_DIR} FOUND)
					if (EXISTS ${PATH} AND ${FOUND} EQUAL -1)
						list(APPEND INCLUDE_PATHS ${PATH} ${DEPENDENCIES_INC_PATHS})
					endif()
				endforeach()
                get_include_paths(DEPENDENCIES_INC_PATHS ${TGT})
            endif()
        endif()
    endforeach()
    set(VISITED_TARGETS ${VISITED_TARGETS} PARENT_SCOPE)
	list(REMOVE_DUPLICATES INCLUDE_PATHS)
    set(${OUTPUT_LIST} ${INCLUDE_PATHS} PARENT_SCOPE)
endfunction()

# Function enabling the Core guideline checker from Visual Studio
option(VCL_ENABLE_CORE_GUIDELINE_CHECKER "Enable core guideline checking" OFF)
function(enable_vs_guideline_checker target)
	get_include_paths(target_include_paths ${target})

	if (MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.13)
		# Exclude external headers from static code analysis. According to a remark in
		# https://blogs.msdn.microsoft.com/vcblog/2017/12/13/broken-warnings-theory/
		# only the environment variabl 'CAExcludePath' seems to work.
		target_compile_options(${target} PRIVATE "/experimental:external" "/external:env:CAExcludePath")
	endif()

	set_target_properties(${target} PROPERTIES
		VS_GLOBAL_EnableCppCoreCheck true
		VS_GLOBAL_CodeAnalysisRuleSet CppCoreCheckRules.ruleset
		VS_GLOBAL_CAExcludePath "${target_include_paths}"
		VS_GLOBAL_RunCodeAnalysis true)
		
	# Pass the information about the core guidelines checker to the target
	target_compile_definitions(${target} PRIVATE VCL_CHECK_CORE_GUIDELINES)
endfunction()

# Checks if a target with a given names exists
function(vcl_check_target tgt)
	if(NOT TARGET ${tgt})
		message(FATAL_ERROR " VCL: compiling vcl requires a ${tgt} CMake target in your project")
	endif()
endfunction()
