#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2018 Basil Fierz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
project(vcl.math.benchmark)

# Status message
message(STATUS "Configuring 'vcl.math.benchmarks'")

# Includes
include(VCLConfig)
include(VCLCudaSupport)
include(VCLOpenCLSupport)

# Requirements
vcl_check_target(benchmark)

set(COMMON_FILES
	problems.cpp
	problems.h
)

# Correctness benchmarks for 3x3 Eigen deompositions
set(EIGEN33_CORRECTNESS_SOURCE_FILES
	eigen33_correctness.cpp
)
add_executable(vcl.math.eigen33.correctness "")
vcl_target_sources(vcl.math.eigen33.correctness "" ${COMMON_FILES} ${EIGEN33_CORRECTNESS_SOURCE_FILES})
set_target_properties(vcl.math.eigen33.correctness PROPERTIES FOLDER benchmarks)
target_link_libraries(vcl.math.eigen33.correctness
	vcl_math
)

# Performance benchmarks for 3x3 Eigen deompositions
set(EIGEN33_PERF_SOURCE_FILES
	eigen33_performance.cpp
)
add_executable(vcl.math.eigen33.performance "")
vcl_target_sources(vcl.math.eigen33.performance "" ${COMMON_FILES} ${EIGEN33_PERF_SOURCE_FILES})
set_target_properties(vcl.math.eigen33.performance PROPERTIES FOLDER benchmarks)
target_link_libraries(vcl.math.eigen33.performance
	vcl_math
	benchmark
)

# Correctness benchmarks for 3x3 rotation computation
set(ROTATION33_CORRECTNESS_SOURCE_FILES
	rotation33_correctness.cpp
)
add_executable(vcl.math.rotation33.correctness "")
vcl_target_sources(vcl.math.rotation33.correctness "" ${COMMON_FILES} ${ROTATION33_CORRECTNESS_SOURCE_FILES})
set_target_properties(vcl.math.rotation33.correctness PROPERTIES FOLDER benchmarks)
target_link_libraries(vcl.math.rotation33.correctness
	vcl_math
)

# Performance benchmarks for 3x3 rotation computation
set(ROTATION33_PERF_SOURCE_FILES
	rotation33_performance.cpp
)
add_executable(vcl.math.rotation33.performance "")
vcl_target_sources(vcl.math.rotation33.performance "" ${COMMON_FILES} ${ROTATION33_PERF_SOURCE_FILES})
set_target_properties(vcl.math.rotation33.performance PROPERTIES FOLDER benchmarks)
target_link_libraries(vcl.math.rotation33.performance
	vcl_math
	benchmark
)

# Correctness benchmarks for 3x3 SVDs
set(SVD33_CORRECTNESS_SOURCE_FILES
	svd33_correctness.cpp
)
add_executable(vcl.math.svd33.correctness "")
vcl_target_sources(vcl.math.svd33.correctness "" ${COMMON_FILES} ${SVD33_CORRECTNESS_SOURCE_FILES})
set_target_properties(vcl.math.svd33.correctness PROPERTIES FOLDER benchmarks)
target_link_libraries(vcl.math.svd33.correctness
	vcl_math
)
if(CUDA_FOUND AND VCL_CUDA_SUPPORT)
	target_link_libraries(vcl.math.svd33.correctness
		vcl_math_cuda
	)
endif()
if(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	target_link_libraries(vcl.math.svd33.correctness
		vcl_math_opencl
	)
endif()

# Performance benchmarks for 3x3 SVDs
set(SVD33_PERF_SOURCE_FILES
	svd33_performance.cpp
)
add_executable(vcl.math.svd33.performance "")
vcl_target_sources(vcl.math.svd33.performance "" ${COMMON_FILES} ${SVD33_PERF_SOURCE_FILES})
set_target_properties(vcl.math.svd33.performance PROPERTIES FOLDER benchmarks)
target_link_libraries(vcl.math.svd33.performance
	vcl_math
	benchmark
)
if(CUDA_FOUND AND VCL_CUDA_SUPPORT)
	target_link_libraries(vcl.math.svd33.performance
		vcl_math_cuda
	)
endif()
if(OPENCL_FOUND AND VCL_OPENCL_SUPPORT)
	target_link_libraries(vcl.math.svd33.performance
		vcl_math_opencl
	)
endif()
