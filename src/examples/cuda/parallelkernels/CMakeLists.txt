cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(parallelkernels_cuda LANGUAGES CXX CUDA)

# Define the project's target
add_executable(parallelkernels_cuda "")

# Place the target into a specific solution folder
set_target_properties(parallelkernels_cuda PROPERTIES FOLDER examples/cuda)

# Enable C++11 for CUDA
target_compile_features(parallelkernels_cuda PUBLIC cxx_std_11)

# Use separate compilation and linking for CUDA
set_target_properties(parallelkernels_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Define the compilation targets
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_30,code=sm_30")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=compute_50")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=compute_60")

# Define the source
set(SRC
	main.cpp
	vectoradd.cu
)
source_group("" FILES ${SRC})
target_sources(parallelkernels_cuda PRIVATE ${SRC})

# Link dependent libraries
target_link_libraries(parallelkernels_cuda
	vcl_compute_cuda
)