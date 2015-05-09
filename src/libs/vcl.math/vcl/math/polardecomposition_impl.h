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
#include <vcl/math/jacobisvd33_qr_impl.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Mathematics
{
	/*!
	 *	\note:
	 *	Compute the polar decomposition of a 3x3 matrix. Since this method is mainly used
	 *	in the context of mesh based FEM simulation we use the method showed in
	 *	"Irving, Teran, Fedkiw - Tetrahedral and Hexahedral Invertible Finite Elements" and
	 *	"Sorkine, Alexa - As-Rigid-As-Possible Surface Modeling".
	 *
	 *	The method is based on an SVD of the input matrix. Contrary to the code in Eigen,
	 *	we handle reflections differently.
	 */
	template<typename REAL>
	void PolarDecomposition(Eigen::Matrix<REAL, 3, 3>& A, Eigen::Matrix<REAL, 3, 3>& R, Eigen::Matrix<REAL, 3, 3>* S)
	{
		Eigen::Matrix<REAL, 3, 3> SV = A;
		Eigen::Matrix<REAL, 3, 3> U, V;
		QRJacobiSVD<REAL>(SV, U, V);

		// Adapted the polar decomposition from Eigen
		REAL x = (U * V.transpose()).determinant();
		Check(Vcl::Mathematics::equal(abs(x), 1, (REAL) 1e-5), "Determinant is -1 or 1.");

		// Assumes ordered singular values
		Check(abs(SV(2, 2)) <= abs(SV(1, 1)) && abs(SV(1, 1)) <= abs(SV(0, 0)), "Singular values are ordered");

		// Fix smallest SV
		REAL sign = select(x < REAL(0), REAL(-1), REAL(1));
		V.col(2) *= sign;

		R = U * V.transpose();

		if (S)
		{
			SV(2, 2) *= sign;
			*S = V * SV.diagonal().asDiagonal() * V.transpose();
		}
	}
}}
