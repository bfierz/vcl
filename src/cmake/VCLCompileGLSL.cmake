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

# Path to the glslc compiler
set(VCL_GLSLC_PATH CACHE FILEPATH "Path to glslc")
set(VCLCOMPILEGLSL_DIR ${CMAKE_CURRENT_LIST_DIR})

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
			"${VCL_GLSLC_PATH}" --target-env=${target_env} ${include_dir_param} -o ${tmp_file} ${file_to_compile}
			
		COMMAND
			${CMAKE_COMMAND} ARGS ${bin2c_cmdline}
		
		MAIN_DEPENDENCY
			${file_to_compile}
		
		COMMENT
			"Compiling ${file_to_compile} to ${output_cpp_file} and ${output_h_file}"
	)

endfunction(VclCompileGLSL)
