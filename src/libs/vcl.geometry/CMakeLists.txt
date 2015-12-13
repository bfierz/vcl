PROJECT(vcl_geometry)

# VCL / GEOMETRY
SET(VCL_GEOMETRY_INC
	vcl/geometry/cell.h
	vcl/geometry/distancePoint3Triangle3.h
	vcl/geometry/distanceTriangle3Triangle3.h
	vcl/geometry/genericid.h
	vcl/geometry/intersect.h
	vcl/geometry/meshfactory.h
	vcl/geometry/property.h
	vcl/geometry/propertygroup.h
	vcl/geometry/simplex.h
	vcl/geometry/tetramesh.h
	vcl/geometry/quad.h
	vcl/geometry/ray.h
	vcl/geometry/segment.h
	vcl/geometry/sphere.h
	vcl/geometry/tetrahedron.h
	vcl/geometry/triangle.h
)
SET(VCL_GEOMETRY_SRC
	vcl/geometry/distancePoint3Triangle3.cpp
	vcl/geometry/distanceTriangle3Triangle3.cpp
	vcl/geometry/intersect.cpp
	vcl/geometry/meshfactory.cpp
	vcl/geometry/tetramesh.cpp
)

SOURCE_GROUP("" FILES ${VCL_GEOMETRY_SRC} ${VCL_GEOMETRY_INC})

SET(SOURCE
	${VCL_GEOMETRY_SRC} ${VCL_GEOMETRY_INC}
)

# Generate library
ADD_LIBRARY(vcl_geometry STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_geometry PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

TARGET_LINK_LIBRARIES(vcl_geometry
	vcl_core
)