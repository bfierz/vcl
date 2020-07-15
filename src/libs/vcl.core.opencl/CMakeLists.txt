PROJECT(vcl_core_opencl)

INCLUDE(${CMAKE_SOURCE_DIR}/src/cmake/VCLCompileCL.cmake)

# VCL / CORE / OPENCL
SET(VCL_CORE_OPENCL_INC
	./vcl/core/opencl/radixsort.h
	./vcl/core/opencl/scan.h
)
SET(VCL_CORE_OPENCL_SRC
	./vcl/core/opencl/radixsort.cpp
	./vcl/core/opencl/scan.cpp
)
SET(VCL_CORE_OPENCL_CL
	vcl/core/opencl/radixsort.cl
	vcl/core/opencl/scan.cl
)

# Access the include directories in order to compile the OpenCL code
GET_PROPERTY(CURR_INC_DIRS TARGET vcl_core PROPERTY INCLUDE_DIRECTORIES)
LIST(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

VCLCOMPILECL(
	${PROJECT_SOURCE_DIR}/vcl/core/opencl/radixsort.cl
	"RadixSortCL"
	"${CURR_INC_DIRS}"
	RADIXSORT_COMPILEDKERNELS
)
VCLCOMPILECL(
	${PROJECT_SOURCE_DIR}/vcl/core/opencl/scan.cl
	"ScanCL"
	"${CURR_INC_DIRS}"
	SCAN_COMPILEDKERNELS
)

SOURCE_GROUP("clc" FILES ${RADIXSORT_COMPILEDKERNELS} ${SCAN_COMPILEDKERNELS})
SOURCE_GROUP("" FILES ${VCL_CORE_OPENCL_SRC} ${VCL_CORE_OPENCL_INC} ${VCL_CORE_OPENCL_CL})

SET(SOURCE
	${RADIXSORT_COMPILEDKERNELS} ${SCAN_COMPILEDKERNELS}
	${VCL_CORE_OPENCL_SRC} ${VCL_CORE_OPENCL_INC}
	${VCL_CORE_OPENCL_CL}
)

# Generate library
ADD_LIBRARY(vcl_core_opencl STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_core_opencl PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_core_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add dependencies
TARGET_LINK_LIBRARIES(vcl_core_opencl vcl_core vcl_compute_opencl)
