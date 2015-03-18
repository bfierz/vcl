PROJECT(cuc)

# Status message
MESSAGE(STATUS "Configuring 'cuc'")

# Boost dependency
FIND_PACKAGE(Boost 1.57 COMPONENTS program_options REQUIRED)
MESSAGE(STATUS "* Boost Include: ${Boost_INCLUDE_DIR}")
MESSAGE(STATUS "* Boost Libraries: ${Boost_LIBRARY_DIRS}")
MESSAGE(STATUS "* Boost Libraries: ${Boost_LIBRARIES}")

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

# Include dependencies
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

# Generate library
ADD_EXECUTABLE(cuc ${SOURCE})
SET_TARGET_PROPERTIES(cuc PROPERTIES FOLDER tools)

TARGET_LINK_LIBRARIES(cuc
	vcl_core
	${Boost_LIBRARIES}
)
