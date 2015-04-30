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

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/contract.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Mathematics
{
	template<typename T>
	struct TwoSidedJacobiTraits {};

	template<>
	struct TwoSidedJacobiTraits<float>
	{
		VCL_STRONG_INLINE static int maxIterations() { return 20; }

		// 2^-22 (Machine eps: 2^-23)
		VCL_STRONG_INLINE static float epsilon() { return 0.0000002384185791015625f; }
	};

	template<>
	struct TwoSidedJacobiTraits<float4>
	{
		VCL_STRONG_INLINE static int maxIterations() { return 20; }

		// 2^-22 (Machine eps: 2^-23)
		VCL_STRONG_INLINE static float epsilon() { return 0.0000002384185791015625f; }
	};

	template<>
	struct TwoSidedJacobiTraits<float8>
	{
		VCL_STRONG_INLINE static int maxIterations() { return 20; }

		// 2^-22 (Machine eps: 2^-23)
		VCL_STRONG_INLINE static float epsilon() { return 0.0000002384185791015625f; }
	};

	template<>
	struct TwoSidedJacobiTraits<float16>
	{
		VCL_STRONG_INLINE static int maxIterations() { return 20; }

		// 2^-22 (Machine eps: 2^-23)
		VCL_STRONG_INLINE static float epsilon() { return 0.0000002384185791015625f; }
	};

	template<>
	struct TwoSidedJacobiTraits<double>
	{
		VCL_STRONG_INLINE static int maxIterations() { return 20; }

		// 2^-51 (Machine eps: 2^-52)
		VCL_STRONG_INLINE static double epsilon() { return 4.4408920985006261616945266723633e-16; }
	};

	// Forsythe and Henrici
	// Not favourable. May yield negative singular values.
	template<typename Real>
	VCL_STRONG_INLINE Eigen::Matrix<Real, 4, 1> TwoSidedJacobiRotationAngleFH(const Real& a11, const Real& a12, const Real& a21, const Real& a22, Real& d1, Real& d2)
	{
		Real X1, X2;
		Real S1, S2;

		Real u11 = a22 - a11;
		Real u12 = a21 + a12;

		Real u21 = a22 + a11;
		Real u22 = a21 - a12;

		if (std::abs(u12) < Real(1e-6)*std::abs(u11))
		{
			X1 = 1;
			S1 = 0;
		}
		else
		{
			// Exact arithmetic operations
			Real rho = u11 / u12;
			Real tau = 1 / (std::abs(rho) + sqrt(1 + rho*rho));
			tau = (rho < 0) ? -tau : tau;
			
			X1 = 1 / sqrt(1 + tau*tau);
			S1 = X1 * tau;
		}
		
		if (std::abs(u22) < Real(1e-6)*std::abs(u21))
		{
			X2 = 1;
			S2 = 0;
		}
		else
		{
			// Exact arithmetic operations
			Real rho = u21 / u22;
			Real tau = 1 / (std::abs(rho) + sqrt(1 + rho*rho));
			tau = (rho < 0) ? -tau : tau;
			
			X2 = 1 / sqrt(1 + tau*tau);
			S2 = X2 * tau;
		}

		Real c1 = X1*X2 + S1*S2;
		Real s1 = S1*X2 - X1*S2;
		Real c2 = X1*X2 - S1*S2;
		Real s2 = S1*X2 + X1*S2;
		
		// Compute the diagonal of the reduced 2x2 SVD
		d1 = a11*c1*c2 - a21*s1*c2 - a12*c1*s2 + a22*s1*s2;
		d2 = a11*s1*s2 + a21*c1*s2 + a12*s1*c2 + a22*c1*c2;

		return Eigen::Matrix<Real, 4, 1>(c1, s1, c2, s2);
	}

	// Brent, Luk, van Loan
	template<typename Real>
	VCL_STRONG_INLINE Eigen::Matrix<Real, 4, 1> TwoSidedJacobiRotationAngleBLL(const Real& a11_in, const Real& a12_in, const Real& a21_in, const Real& a22, Real& d1, Real& d2)
	{
		// Rotation angles
		Real c, s;
		Real c1, s1, c2, s2;

		// Intermediate values
		Real rho, tau;
		Real u1, u2;
		
		// Input values
		Real a11 = a11_in;
		Real a12 = a12_in;
		Real a21 = a21_in;

		//bool flag = (abs(a21) < 1e-6 && abs(a22) < 1e-6);
		//if (flag)
		//{
		//	a21 = a12;
		//	a12 =   0;
		//}
		auto b0 = (abs(a21) < Real(1e-6)) && (abs(a22) < Real(1e-6));
		a21 = select(b0,     a12, a21);
		a12 = select(b0, Real(0), a12);

		u1 = a11 + a22;
		u2 = a12 - a21;
		
		//if ((abs(u1) < 1e-6 && abs(u2) < 1e-6) || (abs(u2) < Real(1e-6)*abs(u1)))
		//{
		//	c = 1;
		//	s = 0;
		//}
		//else
		//{
		//	Real rho = u1 / u2;
		//	s = Real(1) / sqrt(1 + rho*rho);
		//	s = (rho < 0) ? -s : s;
		//	c = s*rho;
		//}
		rho = u1 / u2;
		s = Real(1) / sqrt(Real(1) + rho*rho);
		s = select(rho < 0, -s, s);
		c = s*rho;
		
		auto b1 = ((abs(u1) < Real(1e-6)) && (abs(u2) < Real(1e-6))) || (abs(u2) < Real(1e-6)*abs(u1));
		c = select(b1, Real(1), c);
		s = select(b1, Real(0), s);

		u1 = s*(a12 + a21) + c*(a22 - a11);
		u2 = Real(2)*(c*a12 - s*a22);
		
		//if ((abs(u1) < 1e-6 && abs(u2) < 1e-6) || (abs(u2) < Real(1e-6)*abs(u1)))
		//{
		//	c2 = 1;
		//	s2 = 0;
		//}
		//else
		//{
		//	Real rho = u1 / u2;
		//	Real tau = Real(1) / (abs(rho) + sqrt(Real(1) + rho*rho)); // tau -> tangens
		//	tau = (rho < 0) ? -tau : tau;
		//	c2 = Real(1) / sqrt(Real(1) + tau*tau);
		//	s2 = c2 * tau;
		//}

		// Exact arithmetic operations - tangent
		//rho = u1 / u2;
		//tau = Real(1) / (abs(rho) + sqrt(Real(1) + rho*rho)); // tau -> tangens
		//tau = select(rho < 0, -tau, tau);
		//c2 = Real(1) / sqrt(Real(1) + tau*tau);
		//s2 = c2*tau;
		
		// Optimise by reformulating the cosine or sine:
		// -> http://en.wikipedia.org/wiki/List_of_trigonometric_identities
		// -> csc^2 = 1 + cot^2
		// -> sin   = 1 / csc = 1 / sqrt(1 + cot^2)
		// -> cos   = cot*sin
		
		// Exact arithmetic operations - cotangent
		rho = u1 / u2;
		tau = sgn(rho) * (abs(rho) + sqrt(Real(1) + rho*rho)); // tau -> cotangens
		s2 = Real(1) / sqrt(Real(1) + tau*tau);
		c2 = s2*tau;

		
		auto b2 = ((abs(u1) < Real(1e-6)) && (abs(u2) < Real(1e-6))) || (abs(u2) < Real(1e-6)*abs(u1));
		c2 = select(b2, Real(1), c2);
		s2 = select(b2, Real(0), s2);

		c1 = c2*c - s2*s;
		s1 = s2*c + c2*s;

		// Compute the diagonal of the reduced 2x2 SVD
		d1 = c1*(a11*c2 - a12*s2) - s1*(a21*c2 - a22*s2);
		d2 = s1*(a11*s2 + a12*c2) + c1*(a21*s2 + a22*c2);

		//if (flag)
		//{
		//	c2 = c1;
		//	s2 = s1;
		//	c1 =  1;
		//	s1 =  0;
		//}
		c2 = select(b0,      c1, c2);
		s2 = select(b0,      s1, s2);
		c1 = select(b0, Real(1), c1);
		s1 = select(b0, Real(0), s1);
		
		return Eigen::Matrix<Real, 4, 1>(c1, s1, c2, s2);
	}
	
	template<typename Real>
	VCL_STRONG_INLINE void TwoSidedJacobiRotate(Eigen::Matrix<Real, 3, 3>& A, Eigen::Matrix<Real, 3, 3>& U, Eigen::Matrix<Real, 3, 3>& V, int p, int q, bool normalised)
	{
		Require(0 <= p && p < 3, "p in [0,3)");
		Require(0 <= q && q < 3, "q in [0,3)");
		Require(p < q, "p has to be smaller than q -> (p,q): (0,1),(0,2),(1,2)");
		
		// Rotates A through phi in pq-plane to set A(p, q) = 0 and A(q, p) = 0.
		// Rotation stored in U and V whose columns are orthogonal matrices and U are the eigenvectors of A^T*A.
		Real d1, d2;
		Real kappa(1);
		
		//auto cs = std::move(TwoSidedJacobiRotationAngleFH(A(p, p), A(p, q), A(q, p), A(q, q), d1, d2));
		auto cs = std::move(TwoSidedJacobiRotationAngleBLL(A(p, p), A(p, q), A(q, p), A(q, q), d1, d2));
		Real c1 = cs(0);
		Real s1 = cs(1);
		Real c2 = cs(2);
		Real s2 = cs(3);

		// Normalise result
		if (normalised)
		{
			//if (abs(d2) > abs(d1))
			//{
			//	std::swap(d1, d2);
			//	std::swap(c1, s1); c1 = -c1;
			//	std::swap(c2, s2); c2 = -c2;
			//}
			//if (d1 < 0)
			//{
			//	d1 = -d1;
			//	c1 = -c1;
			//	s1 = -s1;
			//	kappa = -kappa;
			//}
			//if (d2 < 0)
			//{
			//	d2 = -d2;
			//	kappa = -kappa;
			//}
		
			auto b0 = ::abs(d2) > ::abs(d1);
			cswap(b0, d1, d2);
			cswap(b0, c1, s1); c1 = select(b0, -c1, c1);
			cswap(b0, c2, s2); c2 = select(b0, -c2, c2);
		
			auto b1 = d1 < Real(0);
			auto b2 = d2 < Real(0);
			d1 = select(b1, -d1, d1);
			c1 = select(b1, -c1, c1);
			s1 = select(b1, -s1, s1);
			d2 = select(b2, -d2, d2);
		
			kappa = select(b1, -kappa, kappa);
			kappa = select(b2, -kappa, kappa);
		}

		// Sanity check, off diagonal elements must be zero.
#ifdef VCL_DEBUG
		Real Apq = A(p,p)*c1*s2 - A(q,p)*s1*s2 + A(p,q)*c1*c2 - A(q,q)*s1*c2;
		Real Aqp = A(p,p)*kappa*s1*c2 + A(q,p)*kappa*c1*c2 - A(p,q)*kappa*s1*s2 - A(q,q)*kappa*c1*s2;

		Check(all(abs(Apq) < 1e-6), "Off diagonal element is 0.", "Error: %f", Apq);
		Check(all(abs(Aqp) < 1e-6), "Off diagonal element is 0.", "Error: %f", Aqp);

		A(p,q) = Apq;
		A(q,p) = Aqp;
#else
		A(p,q) = 0;
		A(q,p) = 0;
#endif /* VCL_DEBUG */

		A(p,p) = d1;
		A(q,q) = d2;
		
		// Transform A
		int k = 0;
		if (p == 0 && q == 1) k = 2;
		if (p == 0 && q == 2) k = 1;
		if (p == 1 && q == 2) k = 0;
		
		Real Apk = A(p,k)*c1 - A(q,k)*s1;
		Real Aqk = A(p,k)*kappa*s1 + A(q,k)*kappa*c1;
		A(p,k) = Apk;
		A(q,k) = Aqk;

		Real Akp = A(k,p)*c2 - A(k,q)*s2;
		Real Akq = A(k,p)*s2 + A(k,q)*c2;
		A(k,p) = Akp;
		A(k,q) = Akq;

		for (int k = 0; k < 3; k++)
		{
			// Store rotation in U
			Real Ukp = c1 * U(k, p) - s1 * U(k, q);
			Real Ukq = kappa * s1 * U(k, p) + kappa * c1 * U(k, q);
			U(k, p) = Ukp;
			U(k, q) = Ukq;
			
			// Store rotation in V
			Real Vkp = c2 * V(k, p) - s2 * V(k, q);
			Real Vkq = s2 * V(k, p) + c2 * V(k, q);
			V(k, p) = Vkp;
			V(k, q) = Vkq;
		}
	}
	
	template<typename Real, bool Normalised, int p, int q>
	VCL_STRONG_INLINE void TwoSidedJacobiRotate(Eigen::Matrix<Real, 3, 3>& A, Eigen::Matrix<Real, 3, 3>& U, Eigen::Matrix<Real, 3, 3>& V)
	{
		static_assert(0 <= p && p < 3, "p in [0,3)");
		static_assert(0 <= q && q < 3, "q in [0,3)");
		static_assert(p < q, "p has to be smaller than q -> (p,q): (0,1),(0,2),(1,2)");

		TwoSidedJacobiRotate(A, U, V, p, q, Normalised);
	}
	
	// Brent, Luk, van Loan - Computation of the Singular Value Decomposition Using Mesh-Connected Processors
	template<typename Real>
	int TwoSidedJacobiSVD(Eigen::Matrix<Real, 3, 3>& A, Eigen::Matrix<Real, 3, 3>& U, Eigen::Matrix<Real, 3, 3>& V, bool warm_start)
	{
		// A = U S V^T
		// Prepare A in case we use a warm start.
		// Else, initialise U, V to I
		if (warm_start)
		{
			A = U.transpose() * A * V;
		}
		else
		{
			U.setIdentity();
			V.setIdentity();
		}

		// Iterate as long convergence is not reached
		int iter = 0;
		Real err;
		while (iter < TwoSidedJacobiTraits<Real>::maxIterations())
		{
			// Find off diagonal element with maximum modulus
			// Check only upper triangular matrix, assuming that the lower triangular values are handled accordingly
			err = max(abs(A(0,1)), max(abs(A(0,2)), abs(A(1,2))));

			// All small enough -> done
			if (all(err < TwoSidedJacobiTraits<Real>::epsilon()))
				break;
			
			// Rotate matrix with respect to that element
			TwoSidedJacobiRotate<Real, true, 0, 1>(A, U, V);
			iter++;

			err = max(abs(A(0,1)), max(abs(A(0,2)), abs(A(1,2))));
			if (iter >= TwoSidedJacobiTraits<Real>::maxIterations() || all(err < TwoSidedJacobiTraits<Real>::epsilon()))
				break;
			TwoSidedJacobiRotate<Real, true, 0, 2>(A, U, V);
			iter++;

			err = max(abs(A(0,1)), max(abs(A(0,2)), abs(A(1,2))));
			if (iter >= TwoSidedJacobiTraits<Real>::maxIterations() || all(err < TwoSidedJacobiTraits<Real>::epsilon()))
				break;
			TwoSidedJacobiRotate<Real, true, 1, 2>(A, U, V);
			iter++;
		}

		return iter;
	}
}}
