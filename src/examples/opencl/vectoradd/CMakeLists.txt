PROJECT(vectoraddcl)

INCLUDE(${CMAKE_SOURCE_DIR}/cmake/VCLCompileCL.cmake)

SET(SRC
	main.cpp
)

SET(KERNELS
	vectoradd.cl
)

SET(COMPILEDKERNELS
	vectoradd.cl.cpp
)

SOURCE_GROUP("clc" FILES ${COMPILEDKERNELS})
SOURCE_GROUP("" FILES ${SRC} ${KERNELS})

ADD_EXECUTABLE(vectoraddcl ${SRC} ${KERNELS} ${COMPILEDKERNELS})
SET_TARGET_PROPERTIES(vectoraddcl PROPERTIES FOLDER examples/opencl)

TARGET_LINK_LIBRARIES(vectoraddcl
	vcl_compute_opencl
)

# Access the include directories in order to compile the OpenCL code
GET_PROPERTY(CURR_INC_DIRS TARGET vcl_compute_opencl PROPERTY INCLUDE_DIRECTORIES)
LIST(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

VCLCOMPILECL(
	${PROJECT_SOURCE_DIR}/vectoradd.cl
	"vectoradd"
	"${CURR_INC_DIRS}"
	${COMPILEDKERNELS}
)
