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

function(VclCompileCL file_to_compile symbol include_paths compiled_files)

	foreach(dir ${include_paths})
		list(APPEND include_dir_param -I "\"${dir}\"")
	endforeach()

	# Remove the directories from the path and append ".cpp"
	get_filename_component(output_file ${file_to_compile} NAME)
	set(output_file "${output_file}.cpp")
	
	# Append the name to the output
	set(${compiled_files} ${output_file} PARENT_SCOPE)

	add_custom_command(
		OUTPUT ${output_file}

		COMMAND "${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>/clc.exe" --symbol ${symbol} ${include_dir_param} -o ${output_file} ${file_to_compile}
		MAIN_DEPENDENCY ${file_to_compile}
		COMMENT "Compiling ${file_to_compile} to ${output_file}"
	)

endfunction(VclCompileCL)
