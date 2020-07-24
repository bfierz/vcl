#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2019 Basil Fierz
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
project(cui)

# Status message
message(STATUS "Configuring 'cui'")

# Configure libclang
find_path(
	LIBCLANG_INCLUDE_DIR clang-c/Index.h
	REQUIRED
	PATHS
		"C:/Program Files/LLVM/include"
)
find_library(
	LIBCLANG_LIBRARY libclang
	REQUIRED
	HINTS
		"C:/Program Files/LLVM/lib"
)
find_package_handle_standard_args(LIBCLANG
	DEFAULT_MSG
		LIBCLANG_INCLUDE_DIR
		LIBCLANG_LIBRARY
)

# Source code
set(CUI_SRC
	3rdparty/mustache.hpp
	main.cpp
	kernelwrapper.cpp
	kernelwrapper.h
)
source_group("" FILES ${CUI_SRC})

# Generate library
add_executable(cui ${CUI_SRC})
set_target_properties(cui PROPERTIES FOLDER tools)
target_include_directories(cui PRIVATE ${LIBCLANG_INCLUDE_DIR})

target_link_libraries(cui
	vcl_core
	${LIBCLANG_LIBRARY}
)
