PROJECT(vcl_compute)

# VCL / COMPUTE
SET(VCL_COMPUTE_INC
	vcl/compute/buffer.h
	vcl/compute/commandqueue.h
	vcl/compute/context.h
	vcl/compute/module.h
	vcl/compute/kernel.h
		
	#vcl/compute/image.h
	
)
SET(VCL_COMPUTE_SRC
	vcl/compute/buffer.cpp
	vcl/compute/commandqueue.cpp
	vcl/compute/context.cpp
	vcl/compute/module.cpp
	vcl/compute/kernel.cpp
)

SOURCE_GROUP("" FILES ${VCL_COMPUTE_SRC} ${VCL_COMPUTE_INC})

SET(SOURCE
	${VCL_COMPUTE_SRC} ${VCL_COMPUTE_INC}
)

# Generate library
ADD_LIBRARY(vcl_compute STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_compute PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_compute PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

TARGET_LINK_LIBRARIES(vcl_compute
	vcl_core
)