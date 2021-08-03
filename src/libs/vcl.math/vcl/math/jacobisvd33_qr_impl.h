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

// C++ standard library
#include <cmath>

// VCL library
#include <vcl/core/contract.h>
#include <vcl/math/jacobieigen33_selfadjoint_impl.h>
#include <vcl/math/math.h>
#include <vcl/math/qr33_impl.h>

namespace Vcl { namespace Mathematics {
#ifdef VCL_COMPILER_MSVC
#	pragma strict_gs_check(push, off)
#endif // VCL_COMPILER_MSVC

	/*
	 *	Method based on the technical report:
	 *		2011 - McAdams, Selle, Tamstorf, Teran, Sifakis - Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
	 *	which is an extensive description of the method presented in
	 *		SIGGRAPH - 2011 - McAdams, Zhu, Selle, Empey, Tamstorf, Teran, Sifakis - Efficient elasticity for character skinning with contact and collisions
	 */
	template<typename REAL>
	int QRJacobiSVD(Eigen::Matrix<REAL, 3, 3>& A, Eigen::Matrix<REAL, 3, 3>& U, Eigen::Matrix<REAL, 3, 3>& V)
	{
		// Compute eigenvalues of A^T A
		Eigen::Matrix<REAL, 3, 3> ATA = A.transpose() * A;
		int iter_eig33 = SelfAdjointJacobiEigenMaxElement<REAL>(ATA, V);

		// Compute input to QR decomposition
		A *= V;

		// Sort future singular values
		Eigen::Matrix<REAL, 3, 1> b0 = A.col(0);
		Eigen::Matrix<REAL, 3, 1> b1 = A.col(1);
		Eigen::Matrix<REAL, 3, 1> b2 = A.col(2);
		Eigen::Matrix<REAL, 3, 1> v0 = V.col(0);
		Eigen::Matrix<REAL, 3, 1> v1 = V.col(1);
		Eigen::Matrix<REAL, 3, 1> v2 = V.col(2);

		// Column magnitudes
		REAL r0 = A.col(0).squaredNorm();
		REAL r1 = A.col(1).squaredNorm();
		REAL r2 = A.col(2).squaredNorm();

		auto c0 = r0 < r1;
		cnswap(c0, b0, b1);
		cnswap(c0, v0, v1);
		cswap(c0, r0, r1);

		auto c1 = r0 < r2;
		cnswap(c1, b0, b2);
		cnswap(c1, v0, v2);
		cswap(c1, r0, r2);

		auto c2 = r1 < r2;
		cnswap(c2, b1, b2);
		cnswap(c2, v1, v2);
		cswap(c2, r1, r2);

		A.col(0) = b0;
		A.col(1) = b1;
		A.col(2) = b2;
		V.col(0) = v0;
		V.col(1) = v1;
		V.col(2) = v2;

		// Use the QR decomposition to compute U and the singular values
		JacobiQR<REAL>(A, U);

		return iter_eig33 + 3;
	}
#ifdef VCL_COMPILER_MSVC
#	pragma strict_gs_check(pop)
#endif // VCL_COMPILER_MSVC
}}
