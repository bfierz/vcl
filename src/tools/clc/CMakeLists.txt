PROJECT(clc)

# Status message
MESSAGE(STATUS "Configuring 'clc'")

# Boost dependency
FIND_PACKAGE(Boost 1.57 COMPONENTS program_options REQUIRED)
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
ADD_EXECUTABLE(clc ${SOURCE})
SET_TARGET_PROPERTIES(clc PROPERTIES FOLDER tools)

TARGET_LINK_LIBRARIES(clc
	vcl_core
	${Boost_LIBRARIES}
)
