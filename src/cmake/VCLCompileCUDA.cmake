#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2016 Basil Fierz
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

# Path to the vcl cu compiler
SET(VCL_CUC_DIR CACHE PATH "Directory of cuc")

FUNCTION(VclCompileCU file_to_compile symbol include_paths compiled_files)

	FOREACH(dir ${include_paths})
		LIST(APPEND include_dir_param -I "\"${dir}\"")
	ENDFOREACH()

	# Remove the directories from the path and append ".fatbin.cpp"
	GET_FILENAME_COMPONENT(output_file ${file_to_compile} NAME_WE)
	IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
		SET(output_file "${output_file}_m64.fatbin.cpp")
	ELSE()
		SET(output_file "${output_file}_m32.fatbin.cpp")
	ENDIF()
	
	# Append the name to the output
	SET(${compiled_files} ${output_file} PARENT_SCOPE)
	
	# Take cudadevrt appart
	GET_FILENAME_COMPONENT(RT_DIR  ${CUDA_cudadevrt_LIBRARY} DIRECTORY)
	GET_FILENAME_COMPONENT(RT_FILE ${CUDA_cudadevrt_LIBRARY} NAME_WE)	

		# Load old environment if necessary
	SET(VCVARS "")
	IF (MSVC_VERSION EQUAL 1900)
		IF (CMAKE_CL_64)
			SET(VCVARS "C:\\Program Files\\Microsoft SDKs\\Windows\\v7.1\\Bin\\SetEnv.cmd" "/Release" "/x64")
		ELSE(CMAKE_CL_64)
			SET(VCVARS "C:\\Program Files\\Microsoft SDKs\\Windows\\v7.1\\Bin\\SetEnv.cmd" "/Release" "/x86")
		ENDIF(CMAKE_CL_64)
	ENDIF (MSVC_VERSION EQUAL 1900)

	ADD_CUSTOM_COMMAND(
		OUTPUT ${output_file}

		COMMAND ${VCVARS}
		COMMAND "${VCL_CUC_DIR}/cuc.exe" --symbol ${symbol} --profile sm_50 --m64 ${include_dir_param} -L "${RT_DIR}" -l "${RT_FILE}" -o ${output_file} ${file_to_compile}
		MAIN_DEPENDENCY ${file_to_compile}
		COMMENT "Compiling ${file_to_compile} to ${output_file}"
	)

ENDFUNCTION(VclCompileCU)
