PROJECT(vcl_math_opencl)

INCLUDE(${CMAKE_SOURCE_DIR}/cmake/VCLCompileCL.cmake)

# VCL / MATH / OPENCL
SET(VCL_MATH_OPENCL_INC
	vcl/math/opencl/jacobisvd33_mcadams.h
)
SET(VCL_MATH_OPENCL_SRC
	vcl/math/opencl/jacobisvd33_mcadams.cpp
)
SET(VCL_MATH_OPENCL_CL
	vcl/math/opencl/jacobisvd33_mcadams.cl
)

# Access the include directories in order to compile the OpenCL code
GET_PROPERTY(CURR_INC_DIRS TARGET vcl_math PROPERTY INCLUDE_DIRECTORIES)
LIST(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

VCLCOMPILECL(
	${PROJECT_SOURCE_DIR}/vcl/math/opencl/jacobisvd33_mcadams.cl
	"JacobiSVD33McAdamsCL"
	"${CURR_INC_DIRS}"
	COMPILEDKERNELS
)

SOURCE_GROUP("clc" FILES ${COMPILEDKERNELS})
SOURCE_GROUP("" FILES ${VCL_MATH_OPENCL_SRC} ${VCL_MATH_OPENCL_INC} ${VCL_MATH_OPENCL_CL})

SET(SOURCE
	${COMPILEDKERNELS}
	${VCL_MATH_OPENCL_SRC} ${VCL_MATH_OPENCL_INC}
	${VCL_MATH_OPENCL_CL}
)

ADD_LIBRARY(vcl_math_opencl STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_math_opencl PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_math_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add dependencies
TARGET_LINK_LIBRARIES(vcl_math_opencl vcl_core vcl_math vcl_compute_opencl)
