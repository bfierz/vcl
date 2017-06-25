PROJECT(vcl_geometry)

# VCL / GEOMETRY / IO
SET(VCL_GEOMETRY_IO_INC
	vcl/geometry/io/serialiser.h
	
	vcl/geometry/io/tetramesh_serialiser.h	

	vcl/geometry/io/serialiser_nvidia_tet_file.h
	vcl/geometry/io/serialiser_tetgen.h
)
SET(VCL_GEOMETRY_IO_SRC
	vcl/geometry/io/tetramesh_serialiser.cpp

	vcl/geometry/io/serialiser_nvidia_tet_file.cpp
	vcl/geometry/io/serialiser_tetgen.cpp
)

# VCL / GEOMETRY
SET(VCL_GEOMETRY_INC
	vcl/geometry/distancePoint3Triangle3.h
	vcl/geometry/distanceTriangle3Triangle3.h
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

	vcl/geometry/meshfactory.h

	vcl/geometry/simplex.h
	
	vcl/geometry/halffacetetramesh_id.inc
	vcl/geometry/halffacetetramesh.h
	vcl/geometry/multiindextrimesh.h
	vcl/geometry/tetramesh.h
	vcl/geometry/trimesh.h
	
	vcl/geometry/marchingcubestables.h
)
SET(VCL_GEOMETRY_SRC
	vcl/geometry/distancePoint3Triangle3.cpp
	vcl/geometry/distanceTriangle3Triangle3.cpp
	vcl/geometry/intersect.cpp

	vcl/geometry/meshfactory.cpp	
	
	vcl/geometry/halffacetetramesh.cpp
	vcl/geometry/multiindextrimesh.cpp
	vcl/geometry/tetramesh.cpp
	vcl/geometry/trimesh.cpp
	
	vcl/geometry/marchingcubestables.cpp
)

SOURCE_GROUP("io" FILES ${VCL_GEOMETRY_IO_SRC} ${VCL_GEOMETRY_IO_INC})
SOURCE_GROUP("" FILES ${VCL_GEOMETRY_SRC} ${VCL_GEOMETRY_INC})

SET(SOURCE
	${VCL_GEOMETRY_IO_SRC} ${VCL_GEOMETRY_IO_INC}
	${VCL_GEOMETRY_SRC} ${VCL_GEOMETRY_INC}
)

# Generate library
ADD_LIBRARY(vcl_geometry STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_geometry PROPERTIES FOLDER libs)
SET_TARGET_PROPERTIES(vcl_geometry PROPERTIES DEBUG_POSTFIX _d)
TARGET_INCLUDE_DIRECTORIES(vcl_geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

TARGET_LINK_LIBRARIES(vcl_geometry
	vcl_core
)

# Setup installation
INSTALL(FILES ${VCL_GEOMETRY_IO_INC} DESTINATION include/vcl/geometry/io)
INSTALL(FILES ${VCL_GEOMETRY_INC} DESTINATION include/vcl/geometry)

INSTALL(TARGETS vcl_geometry ARCHIVE DESTINATION lib)
