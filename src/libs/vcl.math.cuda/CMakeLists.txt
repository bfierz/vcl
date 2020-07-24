#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2015 Basil Fierz
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
project(vcl_math_cuda LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.9)

# Configure the VCL compilation environment
include(../../cmake/VCLLibrary.cmake)

# Source files
set(VCL_MATH_CUDA_SRC
	vcl/math/cuda/conjugategradientscontext.h
	vcl/math/cuda/conjugategradientscontext.cpp
	vcl/math/cuda/jacobisvd33_mcadams.h
	vcl/math/cuda/jacobisvd33_mcadams.cpp
	vcl/math/solver/cuda/poisson3dsolver_cg.h
	vcl/math/solver/cuda/poisson3dsolver_cg.cpp
	vcl/math/solver/cuda/poisson3dsolver_jacobi.h
	vcl/math/solver/cuda/poisson3dsolver_jacobi.cpp
)
set(VCL_MATH_CUDA_CU
	vcl/math/cuda/conjugategradientscontext.cu
	vcl/math/cuda/jacobisvd33_mcadams.cu
	vcl/math/solver/cuda/poisson3dsolver.cu
	vcl/math/solver/cuda/poisson3dsolver_cg.cu
	vcl/math/solver/cuda/poisson3dsolver_jacobi.cu
)

# Compile CUDA kernels
add_library(vcl_math_cuda_kernels OBJECT "")
vcl_configure(vcl_math_cuda_kernels)
vcl_target_sources(vcl_math_cuda_kernels "vcl/math" ${VCL_MATH_CUDA_CU})
set_target_properties(vcl_math_cuda_kernels PROPERTIES FOLDER libs)
set_target_properties(vcl_math_cuda_kernels PROPERTIES CUDA_PTX_COMPILATION ON)
target_link_libraries(vcl_math_cuda_kernels
	vcl_core
	vcl_core_cuda
	vcl.math
)

get_target_property(cu_files vcl_math_cuda_kernels SOURCES)
foreach(src ${cu_files})
  get_filename_component(src_ext ${src} EXT)
  get_filename_component(src_name ${src} NAME_WE)
  get_filename_component(src_dir ${src} DIRECTORY)

  set(module_file ${CMAKE_CURRENT_BINARY_DIR}/${src_name}_module.cpp)
  list(APPEND module_files ${module_file})
endforeach()

add_custom_command(
  OUTPUT ${module_files}
  COMMAND ${CMAKE_COMMAND}
    "-DCUI_COMMAND=${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>/cui"
    "-DWORK_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DINCLUDE_DIRS=$<TARGET_PROPERTY:vcl_math_cuda_kernels,INCLUDE_DIRECTORIES>"
    "-DSOURCES=${cu_files}"
    "-DOBJECTS=$<TARGET_OBJECTS:vcl_math_cuda_kernels>"
    "-DOUTPUTS=${module_files}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/VclCudaPackPtx.cmake
  VERBATIM
  DEPENDS vcl_math_cuda_kernels
  COMMENT "Generate CUDA kernel C++ bindings"
)


vcl_add_library(vcl_math_cuda)
vcl_target_sources(vcl_math_cuda "vcl/math" ${VCL_MATH_CUDA_SRC})

target_sources(vcl_math_cuda PRIVATE ${module_files})
source_group("generated" FILES ${module_files})

add_dependencies(vcl_math_cuda vcl_math_cuda_kernels)
target_include_directories(vcl_math_cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )

target_link_libraries(vcl_math_cuda
	vcl_core
	vcl.math
	vcl_compute_cuda
)

# Setup installation
install(FILES ${VCL_MATH_CUDA_INC} DESTINATION include/vcl/math/cuda)
install(TARGETS vcl_math_cuda ARCHIVE DESTINATION lib)
