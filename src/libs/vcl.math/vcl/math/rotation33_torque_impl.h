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
#include <vcl/math/math.h>

namespace Vcl { namespace Mathematics
{
	/*
	 *	Method based on:
	 *		2016 - Müller, Bender, Chentanez, Macklin - A Robust Method to Extract the Rotational Part of Deformations
	 */
	template<typename REAL>
	int Rotation(const Eigen::Matrix<REAL, 3, 3>& A, Eigen::Matrix<REAL, 3, 3>& RR)
	{
		const REAL eps = static_cast<REAL>(1e-6f);
		const REAL one = static_cast<REAL>(1);

		Eigen::Quaternion<REAL> q;
		REAL t = RR.trace();
		t = sqrt(t + REAL(1.0));
		q.w() = REAL(0.5)*t;
		t = REAL(0.5) / t;
		q.x() = (RR.coeff(2, 1) - RR.coeff(1, 2)) * t;
		q.y() = (RR.coeff(0, 2) - RR.coeff(2, 0)) * t;
		q.z() = (RR.coeff(1, 0) - RR.coeff(0, 1)) * t;
		q.coeffs() *= one / q.norm();

		int i = 0;
		for (; i < 20; i++)
		{
			Eigen::Matrix<REAL, 3, 3> R = q.matrix();
			Eigen::Matrix<REAL, 3, 1> omega = 
				(R.col(0).cross(A.col(0)) + R.col(1).cross(A.col(1)) + R.col(2).cross(A.col(2)))
				*
				(one / abs(R.col(0).dot(A.col(0)) + R.col(1).dot(A.col(1)) + R.col(2).dot(A.col(2))) + eps);
			REAL w = omega.norm();
			if (all(w < eps))
			{
				RR = R;
				return i + 1;
			}

			q = Eigen::Quaternion<REAL>(Eigen::AngleAxis<REAL>(w, (one / w) * omega)) * q;
			q.coeffs() *= one / q.norm();
		}

		RR = q.matrix();
		return i;
	}
}}
