PROJECT(bin2c)

# Status message
MESSAGE(STATUS "Configuring 'bin2c'")

# Source code
SET(VCL_CLC_INC
)
SET(VCL_CLC_SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${VCL_CLC_SRC} ${VCL_CLC_INC})

SET(SOURCE
	${VCL_CLC_SRC} ${VCL_CLC_INC}
)

# Generate library
ADD_EXECUTABLE(bin2c ${SOURCE})
SET_TARGET_PROPERTIES(bin2c PROPERTIES FOLDER tools)

TARGET_LINK_LIBRARIES(bin2c
	vcl_core
)
