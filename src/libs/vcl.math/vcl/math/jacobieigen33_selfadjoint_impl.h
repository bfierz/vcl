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

// VCL library configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <cmath>

// VCL library
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/contract.h>
#include <vcl/math/math.h>


namespace Vcl { namespace Mathematics
{
	template<typename T>
	struct JacobiTraits {};

	template<>
	struct JacobiTraits<float>
	{
		VCL_STRONG_INLINE static int maxIterations() { return 20; }

		// 2^-22 (Machine eps: 2^-23)
		VCL_STRONG_INLINE static float epsilon() { return 0.0000002384185791015625f; }
	};

	template<>
	struct JacobiTraits<float4>
	{
		VCL_STRONG_INLINE static int maxIterations() { return 20; }

		// 2^-22 (Machine eps: 2^-23)
		VCL_STRONG_INLINE static float epsilon() { return 0.0000002384185791015625f; }
	};

	template<>
	struct JacobiTraits<float8>
	{
		VCL_STRONG_INLINE static int maxIterations() { return 20; }

		// 2^-22 (Machine eps: 2^-23)
		VCL_STRONG_INLINE static float epsilon() { return 0.0000002384185791015625f; }
	};

	template<>
	struct JacobiTraits<double>
	{
		VCL_STRONG_INLINE static int maxIterations() { return 20; }

		// 2^-51 (Machine eps: 2^-52)
		VCL_STRONG_INLINE static double epsilon() { return 4.4408920985006261616945266723633e-16; }
	};
	
	template<typename Real>
	VCL_STRONG_INLINE Eigen::Matrix<Real, 2, 1> JacobiRotationAngle(Real a11, Real a12, Real a22)
	{
		Real u1 = a11 - a22;
		Real u2 = Real(2) * a12;

		// Exact arithmetic operations - tangent
		//Real rho = u1 / u2;
		//Real t = Real(1) / (abs(rho) + sqrt(rho * rho + Real(1))); t -> tangens
		//t = select(rho < 0, -t, t);
		//Real c = Real(1) / sqrt(t * t + 1);
		//Real s = t * c;
		
		// Optimise by reformulating the cosine or sine:
		// -> http://en.wikipedia.org/wiki/List_of_trigonometric_identities
		// -> csc^2 = 1 + cot^2
		// -> sin   = 1 / csc = 1 / sqrt(1 + cot^2)
		// -> cos   = cot*sin

		// Exact arithmetic operations - cotangent
		Real rho = u1 / u2;
		Real ct = sgn(rho) * (abs(rho) + sqrt(rho * rho + Real(1))); // ct -> cotangens
		Real s = Real(1) / sqrt(Real(1) + ct*ct);
		Real c = s*ct;

		// Clamp the angle if it is to large
		auto b = ((abs(u1) < Real(1e-6)) && (abs(u2) < Real(1e-6))) || (abs(u2) < Real(1e-6)*abs(u1));
		c = select(b, Real(1), c);
		s = select(b, Real(0), s);

		return Eigen::Matrix<Real, 2, 1>(c, s);
	}
	
	/*!
	 * Based on McAdams - Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations
	 */
	template<typename Real>
	VCL_STRONG_INLINE Eigen::Matrix<Real, 2, 1> ApproxJacobiRotationAngle(Real a11, Real a12, Real a22)
	{
		auto b = a12*a12 < ((a11-a22)*(a11-a22));
		Real omega = Real(1) / Real(sqrt(a12*a12 + (a11-a22)*(a11-a22)));

		Real s = select(b, omega*a12,       Real(sqrt(0.5)));
		Real c = select(b, omega*(a11-a22), Real(sqrt(0.5)));

		return Eigen::Matrix<Real, 2, 1>(c, s);
	}
	
	template<typename Real, int p, int q>
	void JacobiRotateGeneric(Eigen::Matrix<Real, 3, 3>& R, Eigen::Matrix<Real, 3, 3>& Q)
	{
		static_assert(0 <= p && p < 3, "p in [0,3)");
		static_assert(0 <= q && q < 3, "q in [0,3)");
		static_assert(p < q, "p has to be smaller than q -> (p,q): (0,1),(0,2),(1,2)");

		auto cs = JacobiRotationAngle(R(p, p), R(p, q), R(q, q));
		Real c = cs(0);
		Real s = cs(1);

		Eigen::Matrix<Real, 3, 3> G = Eigen::Matrix<Real, 3, 3>::Identity();
		G(p, p) =  c; G(p, q) = s;
		G(q, p) = -s; G(q, q) = c;

		Q = Q * G;
		R = G.transpose() * R * G;
	}
	
	template<typename Real>
	void JacobiRotateGeneric(Eigen::Matrix<Real, 3, 3>& R, Eigen::Matrix<Real, 3, 3>& Q, int p, int q)
	{
		Require(0 <= p && p < 3, "p in [0,3)");
		Require(0 <= q && q < 3, "q in [0,3)");
		Require(p < q, "p has to be smaller than q -> (p,q): (0,1),(0,2),(1,2)");

		auto cs = JacobiRotationAngle(R(p, p), R(p, q), R(q, q));
		Real c = cs(0);
		Real s = cs(1);

		Eigen::Matrix<Real, 3, 3> G = Eigen::Matrix<Real, 3, 3>::Identity();
		G(p, p) =  c; G(p, q) = s;
		G(q, p) = -s; G(q, q) = c;

		Q = Q * G;
		R = G.transpose() * R * G;
	}

	template<typename REAL>
	VCL_STRONG_INLINE void JacobiRotateOptimised(Eigen::Matrix<REAL, 3, 3>& M, Eigen::Matrix<REAL, 3, 3>& R, int p, int q)
	{
		Require(0 <= p && p < 3, "p in [0,3)");
		Require(0 <= q && q < 3, "q in [0,3)");
		Require(p < q, "p has to be smaller than q -> (p,q): (0,1),(0,2),(1,2)");
		
		// Rotates A through phi in pq-plane to set A(p, q) = 0.
		// Rotation stored in R whose columns are eigenvectors of A
		auto cs = JacobiRotationAngle(M(p, p), M(p, q), M(q, q));
		//auto cs = ApproxJacobiRotationAngle(M(p, p), M(p, q), M(q, q));
		REAL c = cs(0);
		REAL s = cs(1);

		// Compute rotation using sine/cosine
		REAL App = c*c * M(p, p) + REAL(2)*c*s * M(p, q) + s*s * M(q, q);
		REAL Aqq = s*s * M(p, p) - REAL(2)*c*s * M(p, q) + c*c * M(q, q);

#ifdef VCL_DEBUG
		REAL Apq = (c*c-s*s) * M(p, q) - s*c * (M(p, p) - M(q, q));

		Check(all(abs(Apq) < REAL(1e-6)), "Off diagonal element is 0.", "Error: %f", Apq);

		M(p,q) = Apq;
		M(q,p) = Apq;
#else
		M(p,q) = 0;
		M(q,p) = 0;
#endif /* VCL_DEBUG */

		M(p, p) = App;
		M(q, q) = Aqq;

		for (int k = 0; k < 3; k++)
		{
			// Transform A
			if (k != p && k != q)
			{
				REAL Akp = c * M(k, p) + s * M(k, q);
				REAL Akq =-s * M(k, p) + c * M(k, q);
				M(k, p) = M(p, k) = Akp;
				M(k, q) = M(q, k) = Akq;
			}

			// Store rotation in R
			REAL Rkp = c * R(k, p) + s * R(k, q);
			REAL Rkq =-s * R(k, p) + c * R(k, q);
			R(k, p) = Rkp;
			R(k, q) = Rkq;
		}
	}

	template<typename REAL, int p, int q>
	VCL_STRONG_INLINE void JacobiRotateOptimised(Eigen::Matrix<REAL, 3, 3>& A, Eigen::Matrix<REAL, 3, 3>& R)
	{
		static_assert(0 <= p && p < 3, "p in [0,3)");
		static_assert(0 <= q && q < 3, "q in [0,3)");
		static_assert(p < q, "p has to be smaller than q -> (p,q): (0,1),(0,2),(1,2)");

		JacobiRotateOptimised<REAL>(A, R, p, q);
	}
	
	template<typename Real>
	int SelfAdjointJacobiEigenSweeps(Eigen::Matrix<Real, 3, 3>& A, Eigen::Matrix<Real, 3, 3>& R, bool warm_start = false)
	{
		// Initialize the rotation matrix
		if (warm_start)
		{
			A = R.transpose() * A * R;
		}
		else
		{
			R.setIdentity();
		}

		// Only for symmetric matrices!
		// A = R A' R^T, where A' is diagonal and R orthonormal
		// Iterate as long convergence is not reached
	
		// Sweeping over all elements repeatedly
		int iter = 0;
		while (iter < JacobiTraits<Real>::maxIterations())
		{
			// Find off diagonal element with maximum modulus
			// Check only upper triangular matrix, assuming that the lower triangular values are handled accordingly
			Real err = max(abs(A(0,1)), max(abs(A(0,2)), abs(A(1,2))));
		
			// All small enough -> done
			if (all(err < JacobiTraits<Real>::epsilon()))
				break;
			
			// Rotate matrix with respect to that element
			JacobiRotateOptimised<Real, 0, 1>(A, R);
			iter++;
		
			err = max(abs(A(0,1)), max(abs(A(0,2)), abs(A(1,2))));
			if (iter >= JacobiTraits<Real>::maxIterations() || all(err < JacobiTraits<Real>::epsilon()))
				break;
			JacobiRotateOptimised<Real, 0, 2>(A, R);
			iter++;
		
			err = max(abs(A(0,1)), max(abs(A(0,2)), abs(A(1,2))));
			if (iter >= JacobiTraits<Real>::maxIterations() || all(err < JacobiTraits<Real>::epsilon()))
				break;
			JacobiRotateOptimised<Real, 1, 2>(A, R);
			iter++;
		}

		return iter;
	}
	
	template<typename Real>
	int SelfAdjointJacobiEigenMaxElement(Eigen::Matrix<Real, 3, 3>& A, Eigen::Matrix<Real, 3, 3>& R, bool warm_start = false)
	{
		// Initialize the rotation matrix
		if (warm_start)
		{
			A = R.transpose() * A * R;
		}
		else
		{
			R.setIdentity();
		}

		// Only for symmetric matrices!
		// A = R A' R^T, where A' is diagonal and R orthonormal
		// Iterate as long convergence is not reached

		// Always largest remaining element
		int iter = 0;
		while (iter < JacobiTraits<Real>::maxIterations())	// 3 off diagonal elements
		{
			// Find off diagonal element with maximum modulus
			int k = 0;
		
			Real max = abs(A(0, 1));
			Real a = abs(A(0, 2));
			if (any(a > max))
			{
				k = 1;
				max = a;
			}
		
			a = abs(A(1, 2));
			if (any(a > max))
			{
				k = 2;
				max = a;
			}
		
			// All small enough -> done
			if(all(max < JacobiTraits<Real>::epsilon()))
				break;
			
			// Rotate matrix with respect to that element
			switch (k)
			{
			case 0:
				JacobiRotateOptimised<Real, 0, 1>(A, R);
				break;
			case 1:
				JacobiRotateOptimised<Real, 0, 2>(A, R);
				break;
			case 2:
				JacobiRotateOptimised<Real, 1, 2>(A, R);
				break;
			}
		
			iter++;
		}

		return iter;
	}
	
	template<typename REAL>
	int SelfAdjointJacobiEigenGeneric(Eigen::Matrix<REAL, 3, 3>& M, Eigen::Matrix<REAL, 3, 3>& R)
	{
		// Initialize the rotation matrix
		R.setIdentity();

		// Only for symmetric matrices!
		// A = R A' R^T, where A' is diagonal and R orthonormal
		int iter = 0;
		while (iter < JacobiTraits<REAL>::maxIterations())	// 3 off diagonal elements
		{
			// Find off diagonal element with maximum modulus
			int k = 0;

			REAL max = abs(M(0, 1));
			REAL a = abs(M(0, 2));
			if (a > max)
			{
				k = 1;
				max = a;
			}

			a = abs(M(1, 2));
			if (a > max)
			{
				k = 2;
				max = a;
			}

			// All small enough -> done
			if (max < JacobiTraits<REAL>::epsilon())
				break;
			
			// Rotate matrix with respect to that element
			switch (k)
			{
			case 0:
				JacobiRotateGeneric<REAL, 0, 1>(M, R);
				break;
			case 1:
				JacobiRotateGeneric<REAL, 0, 2>(M, R);
				break;
			case 2:
				JacobiRotateGeneric<REAL, 1, 2>(M, R);
				break;
			}

			// Alternative: use generic code
			//JacobiRotateGeneric(A, R, p, q);

			iter++;
		}

		return iter;
	}
}}
