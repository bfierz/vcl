PROJECT(clc)

# Status message
MESSAGE(STATUS "Configuring 'clc'")

# Source code
SET(VCL_CLC_INC
	nvidia.h
)
SET(VCL_CLC_SRC
	main.cpp
	nvidia.cpp
)

SOURCE_GROUP("" FILES ${VCL_CLC_SRC} ${VCL_CLC_INC})

SET(SOURCE
	${VCL_CLC_SRC} ${VCL_CLC_INC}
)

# Generate library
ADD_EXECUTABLE(clc ${SOURCE})
SET_TARGET_PROPERTIES(clc PROPERTIES FOLDER tools)

TARGET_LINK_LIBRARIES(clc
	vcl_core
)
