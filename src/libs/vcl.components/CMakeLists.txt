#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2015 Basil Fierz
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
project(vcl.components)

# Configure the VCL compilation environment
include(../../cmake/VCLLibrary.cmake)

# Generate library
vcl_add_library(vcl.components)

set(SOURCE
	vcl/components/componentstore.cpp
	vcl/components/componentstore.h
	vcl/components/entity.h
	vcl/components/entitymanager.cpp
	vcl/components/entitymanager.h
	vcl/components/system.cpp
	vcl/components/system.h
	vcl/components/systemmanager.cpp
	vcl/components/systemmanager.h
)
vcl_target_sources(vcl.components "vcl/components" ${SOURCE})

target_link_libraries(vcl.components
	PUBLIC
		vcl_core
)

# Setup installation
set(INCLUDE_DIR_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(LIBRARY_DIR_PREFIX "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY vcl DESTINATION ${INCLUDE_DIR_PREFIX})
install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION ${LIBRARY_DIR_PREFIX})
