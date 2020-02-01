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
project(vcl.geometry)

# Configure the VCL compilation environment
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/VCLConfig.cmake)

# Generate library
vcl_add_library(vcl.geometry)

set(SOURCE
	vcl/geometry/io/serialiser.h
	
	vcl/geometry/io/tetramesh_serialiser.cpp
	vcl/geometry/io/tetramesh_serialiser.h	
	vcl/geometry/io/serialiser_nvidia_tet_file.cpp
	vcl/geometry/io/serialiser_nvidia_tet_file.h
	vcl/geometry/io/serialiser_tetgen.cpp
	vcl/geometry/io/serialiser_tetgen.h

	vcl/geometry/primitives/obb.h

	vcl/geometry/distance_ray3ray3.cpp
	vcl/geometry/distance_ray3ray3.h
	vcl/geometry/distancePoint3Triangle3.cpp
	vcl/geometry/distancePoint3Triangle3.h
	vcl/geometry/distanceTriangle3Triangle3.cpp
	vcl/geometry/distanceTriangle3Triangle3.h
	vcl/geometry/intersect_tet_tet.cpp
	vcl/geometry/intersect.h

	vcl/geometry/quad.h
	vcl/geometry/ray.h
	vcl/geometry/segment.h
	vcl/geometry/sphere.h
	vcl/geometry/tetrahedron.h
	vcl/geometry/triangle.h

	vcl/geometry/cell.h

	vcl/geometry/genericid.h
	vcl/geometry/property.h
	vcl/geometry/propertygroup.h

	vcl/geometry/meshfactory.cpp
	vcl/geometry/meshfactory.h

	vcl/geometry/simplex.h
	vcl/geometry/multiindextrimesh.cpp
	vcl/geometry/multiindextrimesh.h
	vcl/geometry/tetramesh.cpp
	vcl/geometry/tetramesh.h
	vcl/geometry/trimesh.cpp
	vcl/geometry/trimesh.h
	
	vcl/geometry/marchingcubestables.cpp
	vcl/geometry/marchingcubestables.h

	vcl/geometry/meshoperations.h
)
vcl_target_sources(vcl.geometry "vcl" ${SOURCE})

target_link_libraries(vcl.geometry
	PUBLIC
		vcl_core
)

# Setup installation
set(INCLUDE_DIR_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(LIBRARY_DIR_PREFIX "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY vcl DESTINATION ${INCLUDE_DIR_PREFIX})
install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION ${LIBRARY_DIR_PREFIX})
