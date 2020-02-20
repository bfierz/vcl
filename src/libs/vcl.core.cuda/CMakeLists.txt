PROJECT(vcl_core_cuda)

# VCL / CORE / CUDA
SET(VCL_CORE_CUDA_INC
)
SET(VCL_CORE_CUDA_SRC
)
SET(VCL_CORE_CUDA_CU
	./vcl/core/cuda/common.inc
	./vcl/core/cuda/math.inc
)

SOURCE_GROUP("" FILES ${VCL_CORE_CUDA_SRC} ${VCL_CORE_CUDA_INC} ${VCL_CORE_CUDA_CU})

SET(SOURCE
	${VCL_CORE_CUDA_SRC} ${VCL_CORE_CUDA_INC} ${VCL_CORE_CUDA_CU}
)

# Generate library
ADD_LIBRARY(vcl_core_cuda OBJECT ${SOURCE})
SET_TARGET_PROPERTIES(vcl_core_cuda PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_core_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add dependencies
TARGET_LINK_LIBRARIES(vcl_core_cuda vcl_core vcl_compute_cuda)
