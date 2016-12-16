PROJECT(vcl_math)

# VCL / MATH / MCADAMS
SET(VCL_MATH_MCADAMS_INC
	vcl/math/mcadams/Singular_Value_Decomposition_Givens_QR_Factorization_Kernel.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Jacobi_Conjugation_Kernel.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Kernel_Declarations.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Main_Kernel_Body.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Preamble.hpp
)

# VCL / MATH / SOLVER
SET(VCL_MATH_SOLVER_INC
	vcl/math/solver/conjugategradients.h
	vcl/math/solver/eigenconjugategradientscontext.h
	vcl/math/solver/jacobi.h
	vcl/math/solver/poisson.h
	vcl/math/solver/poisson1dsolver_cg.h
	vcl/math/solver/poisson1dsolver_jacobi.h
	vcl/math/solver/poisson2dsolver_cg.h
	vcl/math/solver/poisson2dsolver_jacobi.h
	vcl/math/solver/poisson3dsolver_cg.h
	vcl/math/solver/poisson3dsolver_jacobi.h
)
SET(VCL_MATH_SOLVER_SRC
	vcl/math/solver/conjugategradients.cpp
	vcl/math/solver/jacobi.cpp
)

# VCL / MATH
SET(VCL_MATH_INC
	vcl/math/jacobieigen33_selfadjoint_impl.h
	vcl/math/jacobieigen33_selfadjoint.h
	vcl/math/jacobieigen33_selfadjoint_quat_impl.h
	vcl/math/jacobieigen33_selfadjoint_quat.h
	vcl/math/jacobisvd33_mcadams.h
	vcl/math/jacobisvd33_qr_impl.h
	vcl/math/jacobisvd33_qr.h
	vcl/math/jacobisvd33_twosided.h
	vcl/math/jacobisvd33_twosided_impl.h
	vcl/math/polardecomposition.h
	vcl/math/polardecomposition_impl.h
	vcl/math/qr33.h
	vcl/math/qr33_impl.h
	vcl/math/waveletstack3d.h

)
SET(VCL_MATH_SRC
	vcl/math/jacobieigen33_selfadjoint.cpp
	vcl/math/jacobieigen33_selfadjoint_quat.cpp
	vcl/math/jacobisvd33_mcadams.cpp
	vcl/math/jacobisvd33_mcadams_sse.cpp
	vcl/math/jacobisvd33_mcadams_avx.cpp
	vcl/math/jacobisvd33_qr.cpp
	vcl/math/jacobisvd33_twosided.cpp
	vcl/math/polardecomposition.cpp
	vcl/math/qr33.cpp
	vcl/math/waveletstack3d.cpp
)

SOURCE_GROUP("McAdams" FILES ${VCL_MATH_MCADAMS_INC})
SOURCE_GROUP("solver" FILES ${VCL_MATH_SOLVER_SRC} ${VCL_MATH_SOLVER_INC})
SOURCE_GROUP("" FILES ${VCL_MATH_SRC} ${VCL_MATH_INC})

SET(SOURCE
	${VCL_MATH_MCADAMS_INC}
	${VCL_MATH_SOLVER_SRC} ${VCL_MATH_SOLVER_INC}
	${VCL_MATH_SRC} ${VCL_MATH_INC}
)

# Generate library
ADD_LIBRARY(vcl_math STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_math PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

TARGET_LINK_LIBRARIES(vcl_math
	vcl_core
)