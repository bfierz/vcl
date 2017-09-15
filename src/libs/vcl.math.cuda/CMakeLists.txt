PROJECT(vcl_math_cuda)

INCLUDE(${CMAKE_SOURCE_DIR}/cmake/VCLCompileCUDA.cmake)

# VCL / MATH / CUDA
SET(VCL_MATH_CUDA_INC
	vcl/math/cuda/jacobisvd33_mcadams.h
	vcl/math/cuda/conjugategradientscontext.h
)
SET(VCL_MATH_CUDA_SRC
	vcl/math/cuda/jacobisvd33_mcadams.cpp
	vcl/math/cuda/conjugategradientscontext.cpp
)
SET(VCL_MATH_CUDA_CU
	vcl/math/cuda/jacobisvd33_mcadams.cu
	vcl/math/cuda/conjugategradientscontext.cu
)

# Access the include directories in order to compile the CUDA code
GET_PROPERTY(CURR_INC_DIRS_0 TARGET vcl_core_cuda PROPERTY INCLUDE_DIRECTORIES)
GET_PROPERTY(CURR_INC_DIRS_1 TARGET vcl_math PROPERTY INCLUDE_DIRECTORIES)
LIST(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CURR_INC_DIRS_0} ${CURR_INC_DIRS_1})

VCLCOMPILECU(
	${PROJECT_SOURCE_DIR}/vcl/math/cuda/conjugategradientscontext.cu
	"CGCtxCU"
	"${CURR_INC_DIRS}"
	COMPILEDKERNELS_0
)
VCLCOMPILECU(
	${PROJECT_SOURCE_DIR}/vcl/math/cuda/jacobisvd33_mcadams.cu
	"JacobiSVD33McAdamsCU"
	"${CURR_INC_DIRS}"
	COMPILEDKERNELS_1
)
SET(COMPILEDKERNELS ${COMPILEDKERNELS_0} ${COMPILEDKERNELS_1})

SOURCE_GROUP("cuc" FILES ${COMPILEDKERNELS})
SOURCE_GROUP("" FILES ${VCL_MATH_CUDA_SRC} ${VCL_MATH_CUDA_INC} ${VCL_MATH_CUDA_CL})

SET(SOURCE
	${COMPILEDKERNELS}
	${VCL_MATH_CUDA_SRC} ${VCL_MATH_CUDA_INC}
	${VCL_MATH_CUDA_CU}
)

ADD_LIBRARY(vcl_math_cuda STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_math_cuda PROPERTIES FOLDER libs)
if(WIN32)
	SET_TARGET_PROPERTIES(vcl_math_cuda PROPERTIES DEBUG_POSTFIX _d)
endif()
TARGET_INCLUDE_DIRECTORIES(vcl_math_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add dependencies
TARGET_LINK_LIBRARIES(vcl_math_cuda
	vcl_core
	vcl_math
	vcl_compute_cuda
)

# Setup installation
INSTALL(FILES ${VCL_MATH_CUDA_INC} DESTINATION include/vcl/math/cuda)
INSTALL(TARGETS vcl_math_cuda ARCHIVE DESTINATION lib)
