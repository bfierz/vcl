#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2020 Basil Fierz
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
set(VCLCOMPILEHLSL_DIR ${CMAKE_CURRENT_LIST_DIR})

if(${CMAKE_VERSION} VERSION_LESS "3.11.0") 
    message(WARNING "Downloading DXC automatically requires CMake 3.11+")
else()

	include(FetchContent)

	FetchContent_Declare(
	  direct_x_shader_compiler
	  URL      https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.5.2010/dxc_2020_10-22.zip
	  URL_HASH SHA256=b691f63778f470ebeb94874426779b2f60685fc8711adf1b1f9f01535d9b67f8
	)
	FetchContent_GetProperties(direct_x_shader_compiler)
	if(NOT direct_x_shader_compiler_POPULATED)
		FetchContent_MakeAvailable(direct_x_shader_compiler)
	endif()

	message(STATUS "Downloaded DXC to ${direct_x_shader_compiler_SOURCE_DIR}")

endif()


# Path to the dxc compiler
set(VCL_DXC_ROOT CACHE PATH "Path to the root directory of the 'Microsoft DirectX Shader Compiler'")

find_program(DXC dxc
	HINTS
		${VCL_DXC_ROOT}
		${direct_x_shader_compiler_SOURCE_DIR}
	PATH_SUFFIXES
		"bin/x64"
	REQUIRED
)

function(VclCompileHLSL file_to_compile profile main_method symbol include_paths compiled_files)

	foreach(dir ${include_paths})
		list(APPEND include_dir_param -I "\"${dir}\"")
	endforeach()

	# Remove the directories from the path and append ".cpp"
	get_filename_component(output_base_file ${file_to_compile} NAME)
	set(tmp_file "${output_base_file}.tmp.cso")
	set(output_cpp_file "${output_base_file}.cso.cpp")
	set(output_h_file "${CMAKE_CURRENT_BINARY_DIR}/${output_base_file}.cso.h")
	
	# Append the name to the output
	set(${compiled_files} ${output_cpp_file} ${output_h_file} PARENT_SCOPE)

	set(bin2c_cmdline
		-DSYMBOL=${symbol}
		-DOUTPUT_C=${output_cpp_file}
		-DOUTPUT_H=${output_h_file}
		-DINPUT_FILE="${tmp_file}"
		-P "${VCLCOMPILEHLSL_DIR}/bin2c.cmake")

	add_custom_command(
		OUTPUT
			${output_cpp_file}
			${output_h_file}

		COMMAND
			"${DXC}" -T ${profile} -E ${main_method} ${include_dir_param} -Fo ${tmp_file} ${file_to_compile}
		COMMAND
			${CMAKE_COMMAND} ARGS ${bin2c_cmdline}
		
		MAIN_DEPENDENCY
			${file_to_compile}
		
		COMMENT
			"Compiling ${file_to_compile} to ${output_cpp_file} and ${output_h_file}"
	)

endfunction(VclCompileHLSL)
