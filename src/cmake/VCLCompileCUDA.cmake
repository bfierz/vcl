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

set(VCL_CUC_NVCC_PATH CACHE FILEPATH "Path to the CUDA compiler to use")

function(VclCompileCU file_to_compile symbol include_paths compiled_files)

	if (CUDA_VERSION_MAJOR LESS 8)
		message(ERROR "Require at least CUDA 8")
		return()
	endif()

	foreach(dir ${include_paths})
		list(APPEND include_dir_param -I "\"${dir}\"")
	endforeach()

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

	# Compiler flag
	set(CUSTOM_NVCC "")
	if (EXISTS "${VCL_CUC_NVCC_PATH}")
		set(CUSTOM_NVCC --nvcc "\"${VCL_CUC_NVCC_PATH}\"")
	endif()

	# Load old environment if necessary
	# CUDA Toolkit 	              | MSVC       | GCC | Clang
	# CUDA 10.1.105               | 1900..1920 |
	# CUDA 10.0.130               | 1900..1916 |
	# CUDA 9.2 (9.2.148 Update 1) | 1900..1913 |
	# CUDA 9.2 (9.2.88)           | 1900..1913 |
	# CUDA 9.1 (9.1.85)           | 1900..1910 |
	# CUDA 9.0 (9.0.76)           | 1900..1910 |
	# CUDA 8.0 (8.0.61 GA2)       | 1900	   |
	# CUDA 8.0 (8.0.44)           | 1900	   |
	set(MSVC_MAX_VERSION MSVC_VERSION)
	if(CUDA_VERSION_MAJOR LESS 9)
		set(MSVC_MAX_VERSION 1900)
	elseif(CUDA_VERSION_MAJOR LESS 10)
		if(CUDA_VERSION_MINOR LESS 2)
			set(MSVC_MAX_VERSION 1910)
		else()
			set(MSVC_MAX_VERSION 1913)
		endif()
	elseif(CUDA_VERSION_MAJOR EQUAL 10)
		if(CUDA_VERSION_MINOR EQUAL 0)
			set(MSVC_MAX_VERSION 1916)
		endif()
	endif()

	set(VCVARS "")
	# Find existing visual studio installations
	# .\vswhere.exe -legacy -prerelease -format json
	if (MSVC_MAX_VERSION EQUAL 1900)
		if(CMAKE_CL_64)
			SET(VCVARS "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin\\amd64\\vcvars64.bat")
		else()
			SET(VCVARS "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin\\vcvars32.bat")
		endif()
	elseif(MSVC_MAX_VERSION LESS 1920)
		file(GLOB compilers RELATIVE "$ENV{VS2017INSTALLDIR}\\VC\\Tools\\MSVC" "$ENV{VS2017INSTALLDIR}\\VC\\Tools\\MSVC/*")
		foreach(compiler ${compilers})
			if(compiler MATCHES "14\.11.+")
				if(CMAKE_CL_64)
					set(VCVARS "$ENV{VS2017INSTALLDIR}\\VC\\Auxiliary\\Build\\vcvars64.bat" -vcvars_ver=14.11)
				else()
					set(VCVARS "$ENV{VS2017INSTALLDIR}\\VC\\Auxiliary\\Build\\vcvars32.bat" -vcvars_ver=14.11)
				endif()
			endif()
		endforeach()
	endif()

	add_custom_command(
		OUTPUT ${output_file}

		COMMAND set curr_dir=%__CD__%
		COMMAND ${VCVARS}
		COMMAND cd /d %curr_dir%
		COMMAND "${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>/cuc.exe" ${CUSTOM_NVCC} --symbol ${symbol} --profile sm_30 --profile sm_50 --profile sm_60 --m64 ${include_dir_param} -L "${RT_DIR}" -l "${RT_FILE}" -o ${output_file} ${file_to_compile}
		MAIN_DEPENDENCY ${file_to_compile}
		COMMENT "Compiling ${file_to_compile} to ${output_file}"
	)

endfunction(VclCompileCU)
