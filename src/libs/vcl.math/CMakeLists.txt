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
project(vcl.math)

# Configure the VCL compilation environment
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/VCLConfig.cmake)

# Generate library
vcl_add_library(vcl.math)

set(SOURCE
	vcl/math/mcadams/Singular_Value_Decomposition_Givens_QR_Factorization_Kernel.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Jacobi_Conjugation_Kernel.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Kernel_Declarations.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Main_Kernel_Body.hpp
	vcl/math/mcadams/Singular_Value_Decomposition_Preamble.hpp

	vcl/math/solver/conjugategradients.cpp
	vcl/math/solver/conjugategradients.h
	vcl/math/solver/eigenconjugategradientscontext.h
	vcl/math/solver/eigenjacobicontext.h
	vcl/math/solver/jacobi.cpp
	vcl/math/solver/jacobi.h
	vcl/math/solver/poisson.h
	vcl/math/solver/poisson1dsolver_cg.h
	vcl/math/solver/poisson1dsolver_jacobi.h
	vcl/math/solver/poisson2dsolver_cg.h
	vcl/math/solver/poisson2dsolver_jacobi.h
	vcl/math/solver/poisson3dsolver_cg.h
	vcl/math/solver/poisson3dsolver_jacobi.h

	vcl/math/apd33.cpp
	vcl/math/apd33.h
	vcl/math/apd33_impl.h
	vcl/math/jacobieigen33_selfadjoint.cpp
	vcl/math/jacobieigen33_selfadjoint.h
	vcl/math/jacobieigen33_selfadjoint_impl.h
	vcl/math/jacobieigen33_selfadjoint_quat.cpp
	vcl/math/jacobieigen33_selfadjoint_quat.h
	vcl/math/jacobieigen33_selfadjoint_quat_impl.h
	vcl/math/jacobisvd33_mcadams.h
	vcl/math/jacobisvd33_mcadams_mat.cpp
	vcl/math/jacobisvd33_mcadams_mat_sse.cpp
	vcl/math/jacobisvd33_mcadams_mat_avx.cpp
	vcl/math/jacobisvd33_mcadams_quat.cpp
	vcl/math/jacobisvd33_mcadams_quat_sse.cpp
	vcl/math/jacobisvd33_mcadams_quat_avx.cpp
	vcl/math/jacobisvd33_qr.cpp
	vcl/math/jacobisvd33_qr.h
	vcl/math/jacobisvd33_qr_impl.h
	vcl/math/jacobisvd33_twosided.cpp
	vcl/math/jacobisvd33_twosided.h
	vcl/math/jacobisvd33_twosided_impl.h
	vcl/math/polardecomposition.cpp
	vcl/math/polardecomposition.h
	vcl/math/polardecomposition_impl.h
	vcl/math/qr33.cpp
	vcl/math/qr33.h
	vcl/math/qr33_impl.h
	vcl/math/rotation33_torque.cpp
	vcl/math/rotation33_torque.h
	vcl/math/rotation33_torque_impl.h
	vcl/math/waveletstack3d.cpp
	vcl/math/waveletstack3d.h
)
vcl_target_sources(vcl.math "vcl" ${SOURCE})

target_link_libraries(vcl.math
	PUBLIC
		vcl_core
)

# Setup installation
set(INCLUDE_DIR_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(LIBRARY_DIR_PREFIX "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY vcl DESTINATION ${INCLUDE_DIR_PREFIX})
install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION ${LIBRARY_DIR_PREFIX})
