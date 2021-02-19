#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2017 Basil Fierz
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
set(VCLCOMPILEGLSL_DIR ${CMAKE_CURRENT_LIST_DIR})

if(${CMAKE_VERSION} VERSION_LESS "3.11.0") 
    message(WARNING "Downloading Shaderc automatically requires CMake 3.11+")
else()

	include(FetchContent)

	# Binary GLSLC releases: https://github.com/google/shaderc/blob/main/downloads.md
	if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
		FetchContent_Declare(
		  glsl_shader_compiler
		  URL      https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/windows/continuous_release_2017/354/20210106-080226/install.zip
		)
	elseif("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Linux")
		FetchContent_Declare(
		  glsl_shader_compiler
		  URL      https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/linux/continuous_clang_release/351/20210106-080034/install.tgz
		)
	endif()
	FetchContent_GetProperties(glsl_shader_compiler)
	if(NOT glsl_shader_compiler_POPULATED)
		FetchContent_Populate(glsl_shader_compiler)
		message(STATUS "Downloaded Shaderc to ${glsl_shader_compiler_SOURCE_DIR}")
	endif()
endif()

# Path to the glslc compiler
set(VCL_SHADERC_ROOT CACHE PATH "Path to the root directory of glslc")

find_program(GLSLC glslc
	HINTS
		${VCL_SHADERC_ROOT}
		${glsl_shader_compiler_SOURCE_DIR}
	PATH_SUFFIXES
		"bin"
	REQUIRED
)

function(VclCompileGLSL file_to_compile target_env symbol include_paths compiled_files)

	foreach(dir ${include_paths})
		list(APPEND include_dir_param -I "\"${dir}\"")
	endforeach()

	# Remove the directories from the path and append ".cpp"
	get_filename_component(output_base_file ${file_to_compile} NAME)
	set(tmp_file "${output_base_file}.tmp.spv")
	set(output_cpp_file "${output_base_file}.spv.cpp")
	set(output_h_file "${CMAKE_CURRENT_BINARY_DIR}/${output_base_file}.spv.h")
	
	# Append the name to the output
	set(${compiled_files} ${output_cpp_file} ${output_h_file} PARENT_SCOPE)

	set(bin2c_cmdline
		-DSYMBOL=${symbol}
		-DOUTPUT_C=${output_cpp_file}
		-DOUTPUT_H=${output_h_file}
		-DINPUT_FILE="${tmp_file}"
		-P "${VCLCOMPILEGLSL_DIR}/bin2c.cmake")

	add_custom_command(
		OUTPUT
			${output_cpp_file}
			${output_h_file}

		COMMAND
			"${GLSLC}" --target-env=${target_env} ${include_dir_param} -o ${tmp_file} ${file_to_compile}
			
		COMMAND
			${CMAKE_COMMAND} ARGS ${bin2c_cmdline}
		
		MAIN_DEPENDENCY
			${file_to_compile}
		
		COMMENT
			"Compiling ${file_to_compile} to ${output_cpp_file} and ${output_h_file}"
	)

endfunction(VclCompileGLSL)
