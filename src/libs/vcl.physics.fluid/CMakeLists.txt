PROJECT(vcl_physics_fluid)

INCLUDE(${CMAKE_SOURCE_DIR}/cmake/VCLCompileCUDA.cmake)

# VCL / PHYSICS / FLUID / CUDA
SET(VCL_PHYSICS_FLUID_CUDA_INC
	./vcl/physics/fluid/cuda/centergrid.inc
	./vcl/physics/fluid/cuda/cudacentergrid.h
	./vcl/physics/fluid/cuda/cudaenergydecomposition.h
	./vcl/physics/fluid/cuda/cudaeulerfluidsimulation.h
	./vcl/physics/fluid/cuda/cudamaccormackadvection.h
	./vcl/physics/fluid/cuda/cudapoisson3dsolver.h
	./vcl/physics/fluid/cuda/cudapoisson3dsolver_jacobi.h
	./vcl/physics/fluid/cuda/cudasemilagrangeadvection.h
	./vcl/physics/fluid/cuda/cudawaveletturbulencefluidsimulation.h
)
SET(VCL_PHYSICS_FLUID_CUDA_CU
	./vcl/physics/fluid/cuda/cudacentergrid.cu
	./vcl/physics/fluid/cuda/cudaenergydecomposition.cu
	./vcl/physics/fluid/cuda/cudaeulerfluidsimulation.cu
	./vcl/physics/fluid/cuda/cudamaccormackadvection.cu
	./vcl/physics/fluid/cuda/cudapoisson3dsolver.cu
	./vcl/physics/fluid/cuda/cudapoisson3dsolver_jacobi.cu
	./vcl/physics/fluid/cuda/cudasemilagrangeadvection.cu
)
SET(VCL_PHYSICS_FLUID_CUDA_SRC
	./vcl/physics/fluid/cuda/cudacentergrid.cpp
	./vcl/physics/fluid/cuda/cudaenergydecomposition.cpp
	./vcl/physics/fluid/cuda/cudaeulerfluidsimulation.cpp
	./vcl/physics/fluid/cuda/cudamaccormackadvection.cpp
	./vcl/physics/fluid/cuda/cudapoisson3dsolver.cpp
	./vcl/physics/fluid/cuda/cudapoisson3dsolver_jacobi.cpp
	./vcl/physics/fluid/cuda/cudasemilagrangeadvection.cpp
	./vcl/physics/fluid/cuda/cudawaveletturbulencefluidsimulation.cpp
)

# VCL / PHYSICS / FLUID / OPENMP
SET(VCL_PHYSICS_FLUID_OPENMP_INC
	./vcl/physics/fluid/openmp/omppoisson3dsolver.h
)
SET(VCL_PHYSICS_FLUID_OPENMP_SRC
	./vcl/physics/fluid/openmp/omppoisson3dsolver.cpp
)

# VCL / PHYSICS / FLUID
SET(VCL_PHYSICS_FLUID_SRC
	./vcl/physics/fluid/centergrid.cpp
	./vcl/physics/fluid/eulerfluidsimulation.cpp
)
SET(VCL_PHYSICS_FLUID_INC
	./vcl/physics/fluid/advection.h
	./vcl/physics/fluid/centergrid.h
	./vcl/physics/fluid/eulerfluidsimulation.h
	./vcl/physics/fluid/poisson3dsolver.h
)

# Access the include directories in order to compile the CUDA code
GET_PROPERTY(CURR_INC_DIRS TARGET vcl_core_cuda PROPERTY INCLUDE_DIRECTORIES)
LIST(APPEND CURR_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

VCLCOMPILECU(
	${PROJECT_SOURCE_DIR}/vcl/physics/fluid/cuda/cudacentergrid.cu
	"CenterGridCudaModule"
	"${CURR_INC_DIRS}"
	COMPILEDKERNELS_0
)
VCLCOMPILECU(
	${PROJECT_SOURCE_DIR}/vcl/physics/fluid/cuda/cudaenergydecomposition.cu
	"EnergyDecompositionCudaModule"
	"${CURR_INC_DIRS}"
	COMPILEDKERNELS_1
)
VCLCOMPILECU(
	${PROJECT_SOURCE_DIR}/vcl/physics/fluid/cuda/cudapoisson3dsolver_jacobi.cu
	"Poisson3dSolverJacobiCudaModule"
	"${CURR_INC_DIRS}"
	COMPILEDKERNELS_2
)
VCLCOMPILECU(
	${PROJECT_SOURCE_DIR}/vcl/physics/fluid/cuda/cudasemilagrangeadvection.cu
	"SemilagrangeAdvectionCudaModule"
	"${CURR_INC_DIRS}"
	COMPILEDKERNELS_3
)
VCLCOMPILECU(
	${PROJECT_SOURCE_DIR}/vcl/physics/fluid/cuda/cudamaccormackadvection.cu
	"MacCormackAdvectionCudaModule"
	"${CURR_INC_DIRS}"
	COMPILEDKERNELS_4
)
VCLCOMPILECU(
	${PROJECT_SOURCE_DIR}/vcl/physics/fluid/cuda/cudapoisson3dsolver.cu
	"Poisson3dSolverCudaModule"
	"${CURR_INC_DIRS}"
	COMPILEDKERNELS_5
)
SET(COMPILEDKERNELS ${COMPILEDKERNELS_0} ${COMPILEDKERNELS_1} ${COMPILEDKERNELS_2} ${COMPILEDKERNELS_3} ${COMPILEDKERNELS_4} ${COMPILEDKERNELS_5})

# Setup folders for the visual studio project
SOURCE_GROUP(cuc FILES ${COMPILEDKERNELS})
SOURCE_GROUP(cuda FILES ${VCL_PHYSICS_FLUID_CUDA_SRC} ${VCL_PHYSICS_FLUID_CUDA_INC} ${VCL_PHYSICS_FLUID_CUDA_CU})
SOURCE_GROUP(openmp FILES ${VCL_PHYSICS_FLUID_OPENMP_SRC} ${VCL_PHYSICS_FLUID_OPENMP_INC})
SOURCE_GROUP("" FILES ${VCL_PHYSICS_FLUID_SRC} ${VCL_PHYSICS_FLUID_INC})

SET(SOURCE
	${VCL_PHYSICS_FLUID_CUDA_SRC} ${VCL_PHYSICS_FLUID_CUDA_INC} ${VCL_PHYSICS_FLUID_CUDA_CU}
	${VCL_PHYSICS_FLUID_OPENMP_SRC} ${VCL_PHYSICS_FLUID_OPENMP_INC}
	${VCL_PHYSICS_FLUID_SRC} ${VCL_PHYSICS_FLUID_INC}
)

# Generate library
ADD_LIBRARY(vcl_physics_fluid STATIC ${SOURCE} ${COMPILEDKERNELS})
SET_TARGET_PROPERTIES(vcl_physics_fluid PROPERTIES FOLDER libs)
SET_TARGET_PROPERTIES(vcl_physics_fluid PROPERTIES DEBUG_POSTFIX _d)
TARGET_INCLUDE_DIRECTORIES(vcl_physics_fluid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_TOOLKIT_INCLUDE} ${GLEW_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(vcl_physics_fluid
	vcl_core
	vcl_math
	vcl_compute
	
	vcl_compute_cuda
	vcl_math_cuda
)

# Setup installation
INSTALL(FILES ${VCL_PHYSICS_FLUID_CUDA_INC} DESTINATION include/vcl/physics/fluid/cuda)
INSTALL(FILES ${VCL_PHYSICS_FLUID_OPENMP_INC} DESTINATION include/vcl/physics/fluid/openmp)
INSTALL(FILES ${VCL_PHYSICS_FLUID_INC} DESTINATION include/vcl/physics/fluid)
INSTALL(TARGETS vcl_physics_fluid ARCHIVE DESTINATION lib)
