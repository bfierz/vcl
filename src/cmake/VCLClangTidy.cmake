#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2018 Basil Fierz
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

# Support for clang-tidy
option(VCL_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)

if(VCL_ENABLE_CLANG_TIDY AND NOT CLANG_TIDY_EXE)
	find_program(
		CLANG_TIDY_EXE
		NAMES "clang-tidy"
		DOC "Path to clang-tidy executable"
	)
	if(NOT CLANG_TIDY_EXE)
		message(STATUS "clang-tidy not found.")
	else()
		message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
		set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-config=")
	endif()
endif()
	
# Enable clang-tidy checking for given target
function(enable_clang_tidy target)
	if (${CMAKE_VERSION} VERSION_LESS "3.6.0") 
		message(ERROR "Clang-tidy integration requires at least CMake 3.6.0")
	endif()
	
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
	if(VCL_ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
		message(STATUS "Enable clang-tidy on ${target}")
		set_target_properties(
			${target} PROPERTIES
			CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
		)
	endif()
endfunction()
