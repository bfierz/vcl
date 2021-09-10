/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// VCL
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl { namespace Mathematics {
	/**
	 *	Original implementation by:
	 *		2011 - McAdams, Selle, Tamstorf, Teran, Sifakis - Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
	 *	which is an extensive description of the method presented in
	 *		SIGGRAPH - 2011 - McAdams, Zhu, Selle, Empey, Tamstorf, Teran, Sifakis - Efficient elasticity for character skinning with contact and collisions
	 *
	 * @brief Numerically approximates a variant of a Singular Value Decomposition A = U*S*V' 
	 * In this variant U and V are rotation matrices without reflections, or equivalently, in the special orthogonal group SO(3)
	 * S is a diagonal matrix with decreasingly ordered diagonal entries S(0,0) >= S(1,1) >= S(2,2), each of which can be negative
	 * In contrast, the common SVD computes a U and V which are orthogonal, so can contain reflections, and the singular values along the diagonal of S are non-negative
	 * @param A inputs the matrix to be decomposed. Outputs the diagonal of S on its diagonal entries. Non-diagonal entries of A are not specified at output
	 * @param U contains the first rotation matrices of the decomposition
	 * @param V contains the third rotation matrices of the decomposition
	 * @param sweeps number of Jacobi sweeps to be executed for numerical approximation. 
	 * 4 has empirically shown to be a good trade-off between speed and accuracy in the paper cited above, while 5 leads to a higher accuracy result around 1e-5 and 1e-6
	 * @returns total number of Givens rotations applied during execution
	 */
	int McAdamsJacobiSVD(Eigen::Matrix<float, 3, 3>& A, Eigen::Matrix<float, 3, 3>& U, Eigen::Matrix<float, 3, 3>& V, unsigned int sweeps);
	inline int McAdamsJacobiSVD(Eigen::Matrix<float, 3, 3>& A, Eigen::Matrix<float, 3, 3>& U, Eigen::Matrix<float, 3, 3>& V) { return McAdamsJacobiSVD(A, U, V, 4); }
	int McAdamsJacobiSVD(Eigen::Matrix<float, 3, 3>& A, Eigen::Quaternion<float>& U, Eigen::Quaternion<float>& V, unsigned int sweeps = 4);
#ifdef VCL_VECTORIZE_SSE
	int McAdamsJacobiSVD(Eigen::Matrix<float4, 3, 3>& A, Eigen::Matrix<float4, 3, 3>& U, Eigen::Matrix<float4, 3, 3>& V, unsigned int sweeps);
	inline int McAdamsJacobiSVD(Eigen::Matrix<float4, 3, 3>& A, Eigen::Matrix<float4, 3, 3>& U, Eigen::Matrix<float4, 3, 3>& V) { return McAdamsJacobiSVD(A, U, V, 4); }
	int McAdamsJacobiSVD(Eigen::Matrix<float4, 3, 3>& A, Eigen::Quaternion<float4>& U, Eigen::Quaternion<float4>& V, unsigned int sweeps = 4);
#endif // defined(VCL_VECTORIZE_SSE)

#ifdef VCL_VECTORIZE_AVX
	int McAdamsJacobiSVD(Eigen::Matrix<float8, 3, 3>& A, Eigen::Matrix<float8, 3, 3>& U, Eigen::Matrix<float8, 3, 3>& V, unsigned int sweeps);
	inline int McAdamsJacobiSVD(Eigen::Matrix<float8, 3, 3>& A, Eigen::Matrix<float8, 3, 3>& U, Eigen::Matrix<float8, 3, 3>& V) { return McAdamsJacobiSVD(A, U, V, false); }
	int McAdamsJacobiSVD(Eigen::Matrix<float8, 3, 3>& A, Eigen::Quaternion<float8>& U, Eigen::Quaternion<float8>& V, unsigned int sweeps = 4);
#endif // defined(VCL_VECTORIZE_AVX)
}}
