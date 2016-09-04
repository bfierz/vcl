PROJECT(vcl_components)

# VCL  / COMPONENTS
SET(VCL_COMPONENTS_INC
	./vcl/components/componentstore.h
	./vcl/components/entity.h
	./vcl/components/entitymanager.h
	./vcl/components/system.h
	./vcl/components/systemmanager.h
)
SET(VCL_COMPONENTS_SRC
	./vcl/components/componentstore.cpp
	./vcl/components/entitymanager.cpp
	./vcl/components/system.cpp
	./vcl/components/systemmanager.cpp
)

SET(SOURCE
	${VCL_COMPONENTS_SRC} ${VCL_COMPONENTS_INC}
)
SOURCE_GROUP("" FILES ${SOURCE})

# Generate library
ADD_LIBRARY(vcl_components STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_components PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_components PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

TARGET_LINK_LIBRARIES(vcl_components
	vcl_core
)
