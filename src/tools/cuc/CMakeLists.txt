PROJECT(cuc)

# Status message
MESSAGE(STATUS "Configuring 'cuc'")

# Source code
SET(VCL_CUC_INC
)
SET(VCL_CUC_SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${VCL_CUC_SRC} ${VCL_CUC_INC})

SET(SOURCE
	${VCL_CUC_SRC} ${VCL_CUC_INC}
)

# Generate library
ADD_EXECUTABLE(cuc ${SOURCE})
SET_TARGET_PROPERTIES(cuc PROPERTIES FOLDER tools)

TARGET_LINK_LIBRARIES(cuc
	vcl_core
)
