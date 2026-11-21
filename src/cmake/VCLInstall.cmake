#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2026 Basil Fierz
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

# Install rules and the exported vclConfig.cmake (docs/adr/0001-conan-with-find-package-seam.md).
#
# All libraries are installed from here, after src/libs was added, so that the library
# CMake files stay free of install logic. Every library is exported as vcl::<name>.
# Libraries that were not configured (graphics backends, CUDA, OpenCL) are skipped.
#
# Usage: include(VCLInstall) once from the top-level project.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library target -> exported name (vcl::<name>)
set(VCL_INSTALL_LIBRARIES
	vcl_core=core
	vcl.math=math
	vcl.geometry=geometry
	vcl.components=components
	vcl_compute=compute
	vcl_graphics=graphics
	vcl_core_opencl=core_opencl
	vcl_math_opencl=math_opencl
	vcl_compute_opencl=compute_opencl
	vcl_compute_cuda=compute_cuda
	vcl_math_cuda=math_cuda
)

set(VCL_INSTALLED_TARGETS)
foreach(entry IN LISTS VCL_INSTALL_LIBRARIES)
	string(REPLACE "=" ";" entry "${entry}")
	list(GET entry 0 tgt)
	list(GET entry 1 export_name)
	if(NOT TARGET ${tgt})
		continue()
	endif()

	set_target_properties(${tgt} PROPERTIES EXPORT_NAME ${export_name})
	add_library(vcl::${export_name} ALIAS ${tgt})
	list(APPEND VCL_INSTALLED_TARGETS ${tgt})

	# Headers sit in <library dir>/vcl/..., which is also the include layout of the install tree
	get_target_property(src_dir ${tgt} SOURCE_DIR)
	if(IS_DIRECTORY "${src_dir}/vcl")
		install(
			DIRECTORY "${src_dir}/vcl"
			DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
			FILES_MATCHING
				PATTERN "*.h"
				PATTERN "*.hpp"
				PATTERN "*.inl"
		)
	endif()
endforeach()

# The configured header of vcl_core is generated into the build tree
get_target_property(vcl_core_binary_dir vcl_core BINARY_DIR)
install(FILES ${vcl_core_binary_dir}/vcl/config/config.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vcl/config)

install(
	TARGETS ${VCL_INSTALLED_TARGETS}
	EXPORT vclTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

set(VCL_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/vcl)
install(
	EXPORT vclTargets
	NAMESPACE vcl::
	DESTINATION ${VCL_INSTALL_CMAKEDIR}
)

configure_package_config_file(
	${CMAKE_CURRENT_LIST_DIR}/vclConfig.cmake.in
	${CMAKE_BINARY_DIR}/vclConfig.cmake
	INSTALL_DESTINATION ${VCL_INSTALL_CMAKEDIR}
)
install(FILES ${CMAKE_BINARY_DIR}/vclConfig.cmake DESTINATION ${VCL_INSTALL_CMAKEDIR})
