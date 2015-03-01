PROJECT(vcl_math)

# VCL / MATH / MCADAMS
SET(VCL_MATH_MCADAMS_INC
	vcl/math/mcadams/Singular_Value_Decomposition_Givens_QR_Factorization_Kernel.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Jacobi_Conjugation_Kernel.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Kernel_Declarations.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Main_Kernel_Body.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Preamble.hpp
)
# VCL / MATH
SET(VCL_MATH_INC
	vcl/math/jacobieigen33_selfadjoint_impl.h
	vcl/math/jacobieigen33_selfadjoint.h
	vcl/math/jacobisvd33_mcadams.h
	vcl/math/jacobisvd33_qr_impl.h
	vcl/math/jacobisvd33_qr.h
	vcl/math/jacobisvd33_twosided.h
	vcl/math/jacobisvd33_twosided_impl.h
	vcl/math/qr33.h
	vcl/math/qr33_impl.h

)
SET(VCL_MATH_SRC
	vcl/math/jacobieigen33_selfadjoint.cpp
	vcl/math/jacobisvd33_mcadams.cpp
	vcl/math/jacobisvd33_mcadams_sse.cpp
	vcl/math/jacobisvd33_mcadams_avx.cpp
	vcl/math/jacobisvd33_qr.cpp
	vcl/math/jacobisvd33_twosided.cpp
	vcl/math/qr33.cpp
)

SOURCE_GROUP("McAdams" FILES ${VCL_MATH_MCADAMS_INC})
SOURCE_GROUP("" FILES ${VCL_MATH_SRC} ${VCL_MATH_INC})

SET(SOURCE
	${VCL_MATH_MCADAMS_INC}
	${VCL_MATH_SRC} ${VCL_MATH_INC}
)

# Generate library
ADD_LIBRARY(vcl_math STATIC ${SOURCE})
SET_TARGET_PROPERTIES(vcl_math PROPERTIES FOLDER libs)
TARGET_INCLUDE_DIRECTORIES(vcl_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

TARGET_LINK_LIBRARIES(vcl_math
	vcl_core
)