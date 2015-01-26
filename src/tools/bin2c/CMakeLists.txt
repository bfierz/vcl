PROJECT(bin2c)

# Status message
MESSAGE(STATUS "Configuring 'bin2c'")

# Boost dependency
FIND_PACKAGE(Boost 1.55 COMPONENTS program_options REQUIRED)
MESSAGE(STATUS "* Boost Include: ${Boost_INCLUDE_DIR}")
MESSAGE(STATUS "* Boost Libraries: ${Boost_LIBRARY_DIRS}")
MESSAGE(STATUS "* Boost Libraries: ${Boost_LIBRARIES}")

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

# Include dependencies
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

# Generate library
ADD_EXECUTABLE(bin2c ${SOURCE})
SET_TARGET_PROPERTIES(bin2c PROPERTIES FOLDER tools)

TARGET_LINK_LIBRARIES(bin2c
	vcl_core
	${Boost_LIBRARIES}
)
