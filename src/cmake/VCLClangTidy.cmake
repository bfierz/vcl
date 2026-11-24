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

# Support for clang-tidy. The checks and the header filter come from .clang-tidy at the
# repository root. Use clang-tidy 18, the version the curated check set is defined for.
option(VCL_ENABLE_CLANG_TIDY "Run clang-tidy on every VCL target while compiling" OFF)

if(VCL_ENABLE_CLANG_TIDY AND NOT CLANG_TIDY_EXE)
	find_program(
		CLANG_TIDY_EXE
		NAMES "clang-tidy-18" "clang-tidy"
		DOC "Path to clang-tidy executable"
	)
	if(NOT CLANG_TIDY_EXE)
		message(FATAL_ERROR "VCL_ENABLE_CLANG_TIDY is set, but clang-tidy was not found. Set CLANG_TIDY_EXE.")
	endif()
	message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
endif()

# Enable clang-tidy checking for given target
function(enable_clang_tidy target)
	if(VCL_ENABLE_CLANG_TIDY)
		set_target_properties(
			${target} PROPERTIES
			CXX_CLANG_TIDY "${CLANG_TIDY_EXE}"
		)
	endif()
endfunction()

# Enable clang-tidy checking for every target compiling sources in 'dir' and its
# subdirectories. Targets with the property VCL_SKIP_CLANG_TIDY set compile third-party
# sources and are left out.
function(enable_clang_tidy_in_directory dir)
	if(NOT VCL_ENABLE_CLANG_TIDY)
		return()
	endif()

	get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
	foreach(target IN LISTS targets)
		get_target_property(type ${target} TYPE)
		get_target_property(skip ${target} VCL_SKIP_CLANG_TIDY)
		if(type MATCHES "^(STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY|EXECUTABLE)$" AND NOT skip)
			enable_clang_tidy(${target})
		endif()
	endforeach()

	get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
	foreach(subdir IN LISTS subdirs)
		enable_clang_tidy_in_directory("${subdir}")
	endforeach()
endfunction()
