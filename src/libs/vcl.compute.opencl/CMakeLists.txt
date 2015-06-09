PROJECT(vcl_compute_opencl)

# VCL / CONFIG
SET(VCL_CONFIG_INC
	vcl/config/opencl.h
)

# VCL / COMPUTE / OPENCL
SET(VCL_COMPUTE_OPENCL_INC
#	vcl/compute/opencl/buffer.h
#	vcl/compute/opencl/commandqueue.h
	vcl/compute/opencl/context.h
	vcl/compute/opencl/device.h
	vcl/compute/opencl/module.h
#	vcl/compute/opencl/kernel.h
	vcl/compute/opencl/platform.h
)
SET(VCL_COMPUTE_OPENCL_SRC
#	vcl/compute/opencl/buffer.cpp
#	vcl/compute/opencl/commandqueue.cpp
	vcl/compute/opencl/context.cpp
	vcl/compute/opencl/device.cpp
	vcl/compute/opencl/module.cpp
#	vcl/compute/opencl/kernel.cpp
	vcl/compute/opencl/platform.cpp
)

SOURCE_GROUP("config" FILES ${VCL_CONFIG_INC})
SOURCE_GROUP("" FILES ${VCL_COMPUTE_OPENCL_SRC} ${VCL_COMPUTE_OPENCL_INC})

SET(SOURCE
	${VCL_CONFIG_INC}
	${VCL_COMPUTE_OPENCL_SRC} ${VCL_COMPUTE_OPENCL_INC}
)

# Generate library
ADD_LIBRARY(vcl_compute_opencl STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_compute_opencl PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_compute_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

TARGET_LINK_LIBRARIES(vcl_compute_opencl
	vcl_core
	vcl_compute
	${OPENCL_LIBRARIES}
)