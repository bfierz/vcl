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

# Determine the compiler vendor
message(STATUS "Detected compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

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
message(STATUS "Using C++${VCL_CXX_STANDARD}")

# Determine platform architecture
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(VCL_ADDRESS_SIZE "64")
else()
	set(VCL_ADDRESS_SIZE "32")
endif()
message(STATUS "Compiling for ${VCL_ADDRESS_SIZE}bit machine")

# Determine the underlying OS
message(STATUS "Running on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# Define vectorisation
set(VCL_VECTORIZE "SSE 4.1" CACHE STRING "Vectorization instruction set")
set_property(CACHE VCL_VECTORIZE PROPERTY STRINGS "Generic" "SSE 2" "SSE 3" "SSSE 3" "SSE 4.1" "SSE 4.2" "AVX" "AVX 2" "AVX 512" "NEON")

if(VCL_VECTORIZE STREQUAL "SSE 2")
	set(VCL_VECTORIZE_SSE2 TRUE)
elseif(VCL_VECTORIZE STREQUAL "SSE 3")
	set(VCL_VECTORIZE_SSE3 TRUE)
elseif(VCL_VECTORIZE STREQUAL "SSSE 3")
	set(VCL_VECTORIZE_SSSE3 TRUE)
elseif(VCL_VECTORIZE STREQUAL "SSE 4.1")
	set(VCL_VECTORIZE_SSE4_1 TRUE)
elseif(VCL_VECTORIZE STREQUAL "SSE 4.2")
	set(VCL_VECTORIZE_SSE4_2 TRUE)
elseif(VCL_VECTORIZE STREQUAL "AVX")
	set(VCL_VECTORIZE_AVX TRUE)
elseif(VCL_VECTORIZE STREQUAL "AVX 2")
	set(VCL_VECTORIZE_AVX2 TRUE)
elseif(VCL_VECTORIZE STREQUAL "AVX 512")
	set(VCL_VECTORIZE_AVX512 TRUE)
elseif(VCL_VECTORIZE STREQUAL "NEON")
	set(VCL_VECTORIZE_NEON TRUE)
endif()
message(STATUS "Compiling for ${VCL_VECTORIZE}")

# Set whether contracts should be used
set(VCL_USE_CONTRACTS CACHE BOOL "Enable contracts")
message(STATUS "Using contracts ${VCL_USE_CONTRACTS}")

# Set whether contracts should be used
option(VCL_ENABLE_COMPILETIME_TRACING "Enable compilation time tracing" OFF)
mark_as_advanced(VCL_ENABLE_COMPILETIME_TRACING)
message(STATUS "Using compilation time tracing ${VCL_ENABLE_COMPILETIME_TRACING}")

# Enable Visual Studio solution folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Configure the compiler options for a VCL target
function(vcl_configure tgt)

	message(STATUS "Configure target: ${tgt}")

	set_target_properties(${tgt} PROPERTIES
		CXX_STANDARD ${VCL_CXX_STANDARD}
		CXX_STANDARD_REQUIRED YES
		CXX_EXTENSIONS NO
	)

	# Control OpenMP support
	option(VCL_OPENMP_SUPPORT "Enable OpenMP support")
	if(VCL_OPENMP_SUPPORT)
		find_package(OpenMP)
		if(OPENMP_FOUND)
			target_compile_options(${tgt} PUBLIC ${OpenMP_CXX_FLAGS})
		endif()
	endif()

	# Configure MSVC compiler
	if(VCL_COMPILER_MSVC)
		# Configure release configuration
		target_compile_options(${tgt} PUBLIC "$<$<CONFIG:RELEASE>:/GS->" "$<$<CONFIG:RELEASE>:/fp:fast>")

		# Configure all configuration
		# * Enable all warnings
		# * Exceptions
		# * RTTI
		# * Don't be permissive
		target_compile_options(${tgt} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>" "$<$<COMPILE_LANGUAGE:CXX>:/GR>")
		target_compile_options(${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W4>")
		if (MSVC_VERSION GREATER 1900)
			target_compile_options(${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/permissive->")
		endif()

		# Make AVX available
		if(VCL_VECTORIZE_AVX512)
			target_compile_options(${tgt} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>")
		elseif(VCL_VECTORIZE_AVX2)
			target_compile_options(${tgt} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>")
		elseif(VCL_VECTORIZE_AVX)
			target_compile_options(${tgt} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>")
		elseif(VCL_VECTORIZE_SSE2 AND VCL_ADDRESS_SIZE EQUAL "32")
			# All x64 bit machine come with SSE2, thus it's defined as default
			target_compile_options(${tgt} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/arch:SSE2>")
		elseif(VCL_VECTORIZE_NEON)
			target_compile_options(${tgt} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/arch:VFPv4>")
		endif()

		# Enable compilation time tracing
		if(${VCL_ENABLE_COMPILETIME_TRACING} AND NOT (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "19.14"))
			target_compile_options(${tgt} PRIVATE "/Bt+" "/d2cgsummary" "/d1reportTime")
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

		if(VCL_VECTORIZE_AVX512)
			target_compile_options(${tgt} PUBLIC "-mavx512f" "-mavx512vl" "-mavx512dq")
		elseif(VCL_VECTORIZE_AVX2)
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
		elseif(VCL_VECTORIZE_NEON)
			if(CMAKE_SIZEOF_VOID_P EQUAL 4)
				target_compile_options(${tgt} PUBLIC "-mfloat-abi=hard" "-mfpu=neon")
			endif()
		endif()
	endif(VCL_COMPILER_GNU OR VCL_COMPILER_CLANG OR VCL_COMPILER_ICC)

	if(VCL_COMPILER_CLANG)
		if("${CMAKE_GENERATOR}" STREQUAL "Visual Studio 16 2019")
			set(VCL_VS_CUSTOM_CLANG_PATH "" CACHE PATH "Path to the custom Clang installation for the Visual Studio ClangCl target")
			if(EXISTS "${VCL_VS_CUSTOM_CLANG_PATH}/bin/clang-cl.exe")
				execute_process(COMMAND ${VCL_VS_CUSTOM_CLANG_PATH}/bin/clang-cl.exe --version OUTPUT_VARIABLE clang_full_version_string)
				string(REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
				message(STATUS "Actual ClangCl version: ${CLANG_VERSION_STRING}")
				if(${VCL_ENABLE_COMPILETIME_TRACING} AND NOT (CLANG_VERSION_STRING VERSION_LESS "9.0.0"))
					target_compile_options(${tgt} PRIVATE "-ftime-trace")
				endif()
				set_target_properties(${tgt} PROPERTIES VS_GLOBAL_LLVMInstallDir ${VCL_VS_CUSTOM_CLANG_PATH})
			endif()
		elseif(${VCL_ENABLE_COMPILETIME_TRACING} AND NOT (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "9.0.0"))
			target_compile_options(${tgt} PRIVATE "-ftime-trace")
		endif()
	endif(VCL_COMPILER_CLANG)
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

# Add files to target
# 'files' are supposed to be defined relative to where the target is defined
function(vcl_target_sources tgt prefix)

	if(NOT "${prefix}" STREQUAL "")
		string(REPLACE "." "\\." prefix ${prefix})
	endif()

	# Configure the VS project file filters
	foreach(file ${ARGN})
		get_filename_component(dir_ "${file}" DIRECTORY)
		if(NOT "${dir_}" STREQUAL "" AND NOT "${prefix}" STREQUAL "")
			# Remove the prefix and an optional '/'
			string(REGEX REPLACE "^${prefix}/?" "" dir_ ${dir_})
			# Replace the path separator with filter separators
			if("${dir_}" STREQUAL "")
				set(dir_ "\\\\")
			else()
				string(REPLACE "/" "\\\\" dir_ ${dir_})
			endif()
		else()
			set(dir_ "\\\\")
		endif()
		source_group(${dir_} FILES ${file})
	endforeach()

	# Add the files to the target
	target_sources(${tgt} PRIVATE ${ARGN})
endfunction()
