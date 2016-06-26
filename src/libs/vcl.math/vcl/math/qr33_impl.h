/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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
#include <vcl/core/contract.h>
#include <vcl/math/math.h>

// Set to use the rsqrt optimization
//#define VCL_MATH_JACOBIQR_USE_RSQRT

namespace Vcl { namespace Mathematics
{
#ifdef VCL_COMPILER_MSVC
#	pragma strict_gs_check(push, off) 
#endif // VCL_COMPILER_MSVC

	template<typename Scalar>
	VCL_STRONG_INLINE Eigen::Matrix<Scalar, 2, 1> JacobiRotationAngle(const Scalar& a11, const Scalar& a21)
	{
		// Normalisation factor
#ifdef VCL_MATH_JACOBIQR_USE_RSQRT
		Scalar rho = rsqrt(a11*a11 + a21*a21);
#else
		Scalar rho = Scalar(1) / Scalar(sqrt(a11*a11 + a21*a21));
#endif // defined(VCL_MATH_JACOBIQR_USE_RSQRT)

		// Rotation angles
		Scalar c = a11 * rho;
		Scalar s = a21 * rho;

		// Check for singular case
		auto b = (a11*a11 + a21*a21) < Scalar(1e-5*1e-5);
		c = select(b, sgn(a11), c);
		s = select(b, Scalar(1), s);

		return Eigen::Matrix<Scalar, 2, 1>(c, s);
	}

	template<typename Scalar, int p, int q>
	VCL_STRONG_INLINE void JacobiRotateQR(Eigen::Matrix<Scalar, 3, 3>& R, Eigen::Matrix<Scalar, 3, 3>& Q)
	{
		static_assert(0 <= p && p < 3, "p in [0,3)");
		static_assert(0 <= q && q < 3, "q in [0,3)");
		static_assert(p > q, "p has to be greater than q -> (p,q): (1,0), (2,0), (2,1)");

		// Rotates A through phi in pq-plane to set R(p, q) = 0.
		// Rotation stored in Q whose columns are eigenvectors of R
		auto cs = JacobiRotationAngle(R(q, q), R(p, q));		
		Scalar c = cs(0);
		Scalar s = cs(1);

		//Eigen::Matrix<Scalar, 3, 3> G = Eigen::Matrix<Scalar, 3, 3>::Identity();
		//G(p, p) = c;  G(q, p) = s;
		//G(p, q) = -s; G(q, q) = c;
		//R = G * R;

		Scalar Rpp = c * R(p, p) - s * R(q, p);
		Scalar Rpq = c * R(p, q) - s * R(q, q);
		Scalar Rqp = s * R(p, p) + c * R(q, p);
		Scalar Rqq = s * R(p, q) + c * R(q, q);
		R(p, p) = Rpp;
		R(p, q) = Rpq;
		R(q, p) = Rqp;
		R(q, q) = Rqq;

		for (int k = 0; k < 3; k++)
		{
			// Transform A
			if (k != p && k != q)
			{
				Scalar Rpk = c * R(p, k) - s * R(q, k);
				Scalar Rqk = s * R(p, k) + c * R(q, k);
				R(p, k) = Rpk;
				R(q, k) = Rqk;
			}

			// Store rotation in Q
			Scalar Qkq = c * Q(k, q) + s * Q(k, p);
			Scalar Qkp =-s * Q(k, q) + c * Q(k, p);
			Q(k, q) = Qkq;
			Q(k, p) = Qkp;
		}
	}

	template<typename Scalar, int c>
	VCL_STRONG_INLINE void HouseholderQR(Eigen::Matrix<Scalar, 3, 3>& R, Eigen::Matrix<Scalar, 3, 3>& Q)
	{
		static_assert(0 <= c && c < 2, "p in [0,2)");

		Eigen::Matrix<Scalar, 3 - c, 1> u = R.block<3 - c, 1>(c, c);
		Scalar s = sgn(u(0)) * u.norm();
		u(0) += s;
		u.normalize();

		auto B = R.block<3 - c, 3 - c>(c, c);
		auto H = Eigen::Matrix<Scalar, 3 - c, 3 - c>::Identity() - Scalar(2) * u * u.transpose();
		B = H * B;

		Eigen::Matrix<Scalar, 3, 3> T = Eigen::Matrix<Scalar, 3, 3>::Identity();
		T.block<3 - c, 3 - c>(c, c) = H;
		Q = Q * T.transpose();
	}

	template<typename Scalar>
	void JacobiQR(Eigen::Matrix<Scalar, 3, 3>& R, Eigen::Matrix<Scalar, 3, 3>& Q)
	{
		// Initialize Q
		Q.setIdentity();

		// Clear values below the diagonal with a fixed sequence (1,0), (2,0), (2,1)
		// of rotations
		JacobiRotateQR<Scalar, 1, 0>(R, Q);
		JacobiRotateQR<Scalar, 2, 0>(R, Q);
		JacobiRotateQR<Scalar, 2, 1>(R, Q);
	}

	template<typename Scalar>
	void HouseholderQR(Eigen::Matrix<Scalar, 3, 3>& R, Eigen::Matrix<Scalar, 3, 3>& Q)
	{
		// Initialize Q
		Q.setIdentity();

		// Clear values below the diagonal with a fixed sequence 0, 1 column elimination
		HouseholderQR<Scalar, 0>(R, Q);
		HouseholderQR<Scalar, 1>(R, Q);
	}
#ifdef VCL_COMPILER_MSVC
#	pragma strict_gs_check(pop) 
#endif // VCL_COMPILER_MSVC
}}
