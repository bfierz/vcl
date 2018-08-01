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

//#define VCL_MATH_SELFADJOINTJACOBI_QUAT_USE_RSQRT
//#define VCL_MATH_SELFADJOINTJACOBI_QUAT_USE_RCP

namespace Vcl { namespace Mathematics
{
	template<typename Real>
	VCL_STRONG_INLINE Eigen::Matrix<Real, 2, 1> ApproxJacobiRotationQuaternion(const Real& a11, const Real& a12, const Real& a22)
	{
		//const Real pi = Real(3.1415926535897932384626433832795);

		//const Real gamma = Real(3) + Real(2) * Real(std::sqrt(Real(2)));
        const Real gamma = typename NumericTrait<Real>::base_t(5.8284271247461900976033774484194);

		//const Real c_star = Real(std::cos(pi / Real(8)));
        const Real c_star = typename NumericTrait<Real>::base_t(0.92387953251128675612818318939679);
		
		//const Real s_star = Real(std::sin(pi / Real(8)));
        const Real s_star = typename NumericTrait<Real>::base_t(0.3826834323650897717284599840304);
		
		// cos(theta / 2)
		Real c_h = Real(2) * (a11 - a22);

		// sin(theta / 2)
		Real s_h = a12;
	
		// Decide which rotation is used
		auto b = gamma*s_h*s_h < c_h*c_h;

		// Normalisation factor
#ifdef VCL_MATH_SELFADJOINTJACOBI_QUAT_USE_RSQRT
		Real omega = rsqrt(s_h*s_h + c_h*c_h);
#else
		Real omega = Real(1) / Real(sqrt(s_h*s_h + c_h*c_h));
#endif // defined(VCL_MATH_SELFADJOINTJACOBI_QUAT_USE_RSQRT)
		
		// Select appropriate rotation angles
		c_h = select(b, omega*c_h, c_star);
		s_h = select(b, omega*s_h, s_star);

		return Eigen::Matrix<Real, 2, 1>(c_h, s_h);
	}

	template<typename Scalar, int p, int q>
	VCL_STRONG_INLINE void QuaternionJacobiRotateIncremental(Eigen::Matrix<Scalar, 3, 3>& M, Eigen::Quaternion<Scalar>& Q)
	{
		// Rotates A through phi in pq-plane to set M(p, q) = 0.
		auto cs = ApproxJacobiRotationQuaternion(M(p, p), M(p, q), M(q, q));
		Scalar c = cs(0);
		Scalar s = cs(1);

		if (p == 0 && q == 1)
		{
			// Define the rotation quaternion
			Eigen::Quaternion<Scalar> quat{ c, 0, 0, s };

#if 0
			// Build the rotation matrix
			Eigen::Matrix<Scalar, 3, 3> R = quat.toRotationMatrix();

			// Transform the matrices
			M = R.transpose() * M * R;
#endif

#if 0
			// Build the rotation matrix
			Eigen::Matrix<Scalar, 3, 3> R;
			R(0, 0) = Scalar(1) - Scalar(2)*s*s;
			R(1, 0) = Scalar(2)*s*c;
			R(2, 0) = 0;

			R(0, 1) = Scalar(-2)*s*c;
			R(1, 1) = Scalar(1) - Scalar(2)*s*s;
			R(2, 1) = 0;

			R(0, 2) = 0;
			R(1, 2) = 0;
			R(2, 2) = Scalar(1);
			
			// Transform the matrices
			M = R.transpose() * M * R;
#endif

#if 1
			// Transform the input matrix
			Scalar a /* = Scalar(1) - Scalar(2)*s*s */ = c*c - s*s;
			Scalar b = Scalar(2)*s*c;
			Scalar M00 = a*a * M(0, 0) + a*b * (M(0, 1) + M(1, 0)) + b*b * M(1, 1);
			Scalar M10 = a*a * M(1, 0) + a*b * (M(1, 1) - M(0, 0)) - b*b * M(0, 1);
			Scalar M20 = a * M(2, 0) + b * M(2, 1);

			//Scalar M01 = a*a * M(0, 1) + a*b * ( M(1, 1) - M(0, 0)) - b*b * M(1, 0);
			Scalar M11 = a*a * M(1, 1) + a*b * (-M(0, 1) - M(1, 0)) + b*b * M(0, 0);
			Scalar M21 = a * M(2, 1) - b * M(2, 0);

			//Scalar M02 = a * M(0, 2) + b * M(1, 2);
			//Scalar M12 = a * M(1, 2) - b * M(0, 2);
			Scalar M22 = M(2, 2);

			M(0, 0) = M00;
			M(1, 0) = M10;
			M(2, 0) = M20;

			M(0, 1) = M(1, 0); //M01;
			M(1, 1) = M11;
			M(2, 1) = M21;

			M(0, 2) = M(2, 0); //M02;
			M(1, 2) = M(2, 1); //M12;
			M(2, 2) = M22;
#endif

			// Update the rotation quaternion
			Q *= quat;
		}
		else if (p == 0 && q == 2)
		{
			// Define the rotation quaternion
			Eigen::Quaternion<Scalar> quat{ c, 0, -s, 0 };

#if 0
			// Build the rotation matrix
			Eigen::Matrix<Scalar, 3, 3> R = quat.toRotationMatrix();

			// Transform the matrices
			M = R.transpose() * M * R;
#endif

#if 0
			// Build the rotation matrix
			Eigen::Matrix<Scalar, 3, 3> R;
			R(0, 0) = Scalar(1) - Scalar(2)*s*s;
			R(1, 0) = 0;
			R(2, 0) = Scalar(2)*s*c;
			
			R(0, 1) = 0;
			R(1, 1) = Scalar(1);
			R(2, 1) = 0;
			
			R(0, 2) = Scalar(-2)*s*c;
			R(1, 2) = 0;
			R(2, 2) = Scalar(1) - Scalar(2)*s*s;

			// Transform the matrices
			M = R.transpose() * M * R;
#endif

#if 1
			// Transform the input matrix
			Scalar a /* = Scalar(1) - Scalar(2)*s*s */ = c*c - s*s;
			Scalar b = Scalar(2)*s*c;
			Scalar M00 = a*a * M(0, 0) + a*b * (M(0, 2) + M(2, 0)) + b*b * M(2, 2);
			Scalar M10 = a * M(1, 0) + b * M(1, 2);
			Scalar M20 = a*a * M(2, 0) + a*b * (M(2, 2) - M(0, 0)) - b*b * M(0, 2);

			//Scalar M01 = a * M(0, 1) + b * M(2, 1);
			Scalar M11 = M(1, 1);
			Scalar M21 = a * M(2, 1) - b * M(0, 1);

			//Scalar M02 = a*a * M(0, 2) + a*b * ( M(2, 2) - M(0, 0)) - b*b * M(2, 0);
			//Scalar M12 = a * M(1, 2) - b * M(1, 0);
			Scalar M22 = a*a * M(2, 2) + a*b * (-M(0, 2) - M(2, 0)) + b*b * M(0, 0);

			M(0, 0) = M00;
			M(1, 0) = M10;
			M(2, 0) = M20;

			M(0, 1) = M(1, 0); //M01;
			M(1, 1) = M11;
			M(2, 1) = M21;

			M(0, 2) = M(2, 0); //M02;
			M(1, 2) = M(2, 1); //M12;
			M(2, 2) = M22;
#endif
			Q *= quat;
		}
		else if (p == 1 && q == 2)
		{
			// Define the rotation quaternion
			Eigen::Quaternion<Scalar> quat{ c, s, 0, 0 };
#if 0
			// Build the rotation matrix
			Eigen::Matrix<Scalar, 3, 3> R = quat.toRotationMatrix();

			// Transform the matrices
			M = R.transpose() * M * R;
#endif

#if 0
			// Build the rotation matrix
			Eigen::Matrix<Scalar, 3, 3> R;
			R(0, 0) = Scalar(1);
			R(1, 0) = 0;
			R(2, 0) = 0;
			
			R(0, 1) = 0;
			R(1, 1) = Scalar(1) - Scalar(2)*s*s;
			R(2, 1) = Scalar(2)*s*c;
			
			R(0, 2) = 0;
			R(1, 2) = Scalar(-2)*s*c;
			R(2, 2) = Scalar(1) - Scalar(2)*s*s;

			// Transform the matrices
			M = R.transpose() * M * R;
#endif

#if 1
			// Transform the input matrix
			Scalar a /* = Scalar(1) - Scalar(2)*s*s */ = c*c - s*s;
			Scalar b = Scalar(2)*s*c;

			Scalar M00 = M(0, 0);
			Scalar M10 = a * M(1, 0) + b * M(2, 0);
			Scalar M20 = a * M(2, 0) - b * M(1, 0);
		
			//Scalar M01 = a * M(0, 1) + b * M(0, 2);
			Scalar M11 = a*a * M(1, 1) + a*b * (M(1, 2) + M(2, 1)) + b*b * M(2, 2);
			Scalar M21 = a*a * M(2, 1) + a*b * (M(2, 2) - M(1, 1)) - b*b * M(1, 2);

			//Scalar M02 = a * M(0, 2) - b * M(0, 1);
			//Scalar M12 = a*a * M(1, 2) + a*b * ( M(2, 2) - M(1, 1)) - b*b * M(2, 1);
			Scalar M22 = a*a * M(2, 2) + a*b * (-M(1, 2) - M(2, 1)) + b*b * M(1, 1);

			M(0, 0) = M00;
			M(1, 0) = M10;
			M(2, 0) = M20;

			M(0, 1) = M(1, 0); //M01;
			M(1, 1) = M11;
			M(2, 1) = M21;

			M(0, 2) = M(2, 0); //M02;
			M(1, 2) = M(2, 1); //M12;
			M(2, 2) = M22;
#endif
			Q *= quat;
		}
	}

	template<typename Scalar, int p, int q>
	VCL_STRONG_INLINE void QuaternionJacobiRotateIncremental(Eigen::Matrix<Scalar, 6, 1>& M, Eigen::Quaternion<Scalar>& Q)
	{
		// Rotates A through phi in pq-plane to set M(p, q) = 0.
		auto cs = ApproxJacobiRotationQuaternion(M(p), M(2 + p + q), M(q));
		Scalar c = cs(0);
		Scalar s = cs(1);

		Scalar a = c*c - s*s;
		Scalar b = Scalar(2)*s*c;
		VCL_IF_CONSTEXPR(p == 0 && q == 1)
		{
			// Define the rotation quaternion
			Eigen::Quaternion<Scalar> quat{ c, 0, 0, s };

			// Transform the input matrix
			Scalar M00 = a*a * M(0) + a*b * (M(3) + M(3)) + b*b * M(1);
			Scalar M10 = a*a * M(3) + a*b * (M(1) - M(0)) - b*b * M(3);
			Scalar M20 = a * M(4) + b * M(5);
			Scalar M11 = a*a * M(1) + a*b * (-M(3) - M(3)) + b*b * M(0);
			Scalar M21 = a * M(5) - b * M(4);

			M(0) = M00;
			M(1) = M11;
			M(3) = M10;
			M(4) = M20;
			M(5) = M21;

			// Update the rotation quaternion
			Q *= quat;
		}
		else VCL_IF_CONSTEXPR(p == 0 && q == 2)
		{
			// Define the rotation quaternion
			Eigen::Quaternion<Scalar> quat{ c, 0, -s, 0 };

			// Transform the input matrix
			Scalar M00 = a*a * M(0) + a*b * (M(4) + M(4)) + b*b * M(2);
			Scalar M10 = a * M(3) + b * M(5);
			Scalar M20 = a*a * M(4) + a*b * (M(2) - M(0)) - b*b * M(4);
			Scalar M21 = a * M(5) - b * M(3);
			Scalar M22 = a*a * M(2) + a*b * (-M(4) - M(4)) + b*b * M(0);

			M(0) = M00;
			M(2) = M22;
			M(3) = M10;
			M(4) = M20;
			M(5) = M21;

			Q *= quat;
		}
		else VCL_IF_CONSTEXPR(p == 1 && q == 2)
		{
			// Define the rotation quaternion
			Eigen::Quaternion<Scalar> quat{ c, s, 0, 0 };

			// Transform the input matrix
			Scalar M10 = a * M(3) + b * M(4);
			Scalar M20 = a * M(4) - b * M(3);
			Scalar M11 = a*a * M(1) + a*b * (M(5) + M(5)) + b*b * M(2);
			Scalar M21 = a*a * M(5) + a*b * (M(2) - M(1)) - b*b * M(5);
			Scalar M22 = a*a * M(2) + a*b * (-M(5) - M(5)) + b*b * M(1);

			M(1) = M11;
			M(2) = M22;
			M(3) = M10;
			M(4) = M20;
			M(5) = M21;

			Q *= quat;
		}
	}

	template<typename Scalar, int p, int q>
	VCL_STRONG_INLINE void QuaternionJacobiRotate(const Eigen::Matrix<Scalar, 3, 3>& M, Eigen::Quaternion<Scalar>& Q)
	{
		VclRequire(all(equal(Q.norm(), Scalar(1), Scalar(1e-6))), "Quaternion is normalized.");
		
		Eigen::Matrix<Scalar, 3, 3> R = Q.toRotationMatrix();
		Eigen::Matrix<Scalar, 3, 3> D = R.transpose() * M * R;

		VCL_IF_CONSTEXPR(p == 0 && q == 1)
		{
			// Rotates A through phi in pq-plane to set D(p, q) = 0.
			auto cs = ApproxJacobiRotationQuaternion(D(0, 0), D(0, 1), D(1, 1));
			Scalar c = cs(0);
			Scalar s = cs(1);

			VclCheck(all(equal(Eigen::Quaternion<Scalar>(c, 0, 0, s).norm(), Scalar(1), Scalar(1e-6))), "Quaternion is normalized.");

			Q *= Eigen::Quaternion<Scalar>(c, 0, 0, s);
		}
		else VCL_IF_CONSTEXPR(p == 0 && q == 2)
		{
			// Rotates A through phi in pq-plane to set D(p, q) = 0.
			auto cs = ApproxJacobiRotationQuaternion(D(0, 0), D(0, 2), D(2, 2));
			Scalar c = cs(0);
			Scalar s = cs(1);

			VclCheck(all(equal(Eigen::Quaternion<Scalar>(c, 0, -s, 0).norm(), Scalar(1), Scalar(1e-6))), "Quaternion is normalized.");

			Q *= Eigen::Quaternion<Scalar>(c, 0, -s, 0);
		}
		else VCL_IF_CONSTEXPR(p == 1 && q == 2)
		{
			// Rotates A through phi in pq-plane to set D(p, q) = 0.
			auto cs = ApproxJacobiRotationQuaternion(D(1, 1), D(1, 2), D(2, 2));
			Scalar c = cs(0);
			Scalar s = cs(1);

			VclCheck(all(equal(Eigen::Quaternion<Scalar>(c, s, 0, 0).norm(), Scalar(1), Scalar(1e-6))), "Quaternion is normalized.");

			Q *= Eigen::Quaternion<Scalar>(c, s, 0, 0);
		}
	}
	
	template<typename Scalar>
	VCL_STRONG_INLINE void QuaternionJacobiRotate(const Eigen::Matrix<Scalar, 3, 3>& M, int p, int q, Eigen::Quaternion<Scalar>& Q)
	{
		VclRequire(equal(Q.norm(), 1, 1e-6), "Quaternion is normalized.");

		Eigen::Matrix<Scalar, 3, 3> R = Q.toRotationMatrix();
		Eigen::Matrix<Scalar, 3, 3> D = R.transpose() * M * R;

		// Rotates A through phi in pq-plane to set D(p, q) = 0.
		auto cs = ApproxJacobiRotationQuaternion(D(p, p), D(p, q), D(q, q));
		Scalar c = cs(0);
		Scalar s = cs(1);

		Eigen::Matrix<Scalar, 3, 1> v = Eigen::Matrix<Scalar, 3, 1>::Zero();
		int idx = 2 - (p + q - 1);
		v(idx) = (idx%2 == 0) ? s : -s;
		
		Q *= Eigen::Quaternion<Scalar>(c, v(0), v(1), v(2));

		VclEnsure(epsEqual(Eigen::Quaternion<Scalar>(c, v(0), v(1), v(2)).norm(), 1, 1e-6), "Quaternion is normalized.");
	}
	
	/*
	 *	Method based on the technical report:
	 *		2011 - McAdams, Selle, Tamstorf, Teran, Sifakis - Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
	 *	which is an extensive description of the method presented in
	 *		SIGGRAPH - 2011 - McAdams, Zhu, Selle, Empey, Tamstorf, Teran, Sifakis - Efficient elasticity for character skinning with contact and collisions
	 */
	template<typename Scalar>
	int SelfAdjointJacobiEigenQuatIncrementalSweeps(Eigen::Matrix<Scalar, 3, 3>& A, Eigen::Quaternion<Scalar>& Q, int nr_sweeps = 5)
	{
		using namespace Eigen;
		
		// Initialize the temporary working set.
		// This informs the optimizer that we are not interested in any
		// intermediate result and only the final result needs to be stored.
		// For this algorithm the compiled code looks much better.
		Eigen::Quaternion<Scalar> U = Eigen::Quaternion<Scalar>::Identity();
#if 0
		Eigen::Matrix<Scalar, 3, 3> M = A;
#else
		Eigen::Matrix<Scalar, 6, 1> M;
		M(0) = A(0, 0);
		M(1) = A(1, 1);
		M(2) = A(2, 2);
		M(3) = A(1, 0);
		M(5) = A(2, 1);
		M(4) = A(2, 0);
#endif

		// Only for symmetric matrices!
		// A = R A' R^T, where A' is diagonal and R orthonormal
		// Use a fixed sequence of operations instead of looking at the largest element
		for (int i = 0; i < nr_sweeps; i++)
		{
			QuaternionJacobiRotateIncremental<Scalar, 0, 1>(M, U);
			QuaternionJacobiRotateIncremental<Scalar, 0, 2>(M, U);
			QuaternionJacobiRotateIncremental<Scalar, 1, 2>(M, U);
		}

		// Normalize and return the rotation quaternion
		//Q = U.normalized();

		// Replace normalization due to Eigen 3.3 compatibility
		Q = U.coeffs() / sqrt(U.coeffs().squaredNorm());

		// Return the Eigenvalues
#if 0
		A = M;
#else
		//A = Eigen::DiagonalMatrix<Scalar, 3>{ M(0), M(1), M(2) };
		A(0, 0) = M(0);
		A(1, 1) = M(1);
		A(2, 2) = M(2);
#endif

		return nr_sweeps * 3;
	}

	template<typename Scalar>
	int SelfAdjointJacobiEigenQuatSweeps(Eigen::Matrix<Scalar, 3, 3>& A, Eigen::Quaternion<Scalar>& Q, int nr_sweeps = 5)
	{
		using namespace Eigen;

		// Initialize Q
		Q = Eigen::Quaternion<Scalar>::Identity();

		// Only for symmetric matrices!
		// A = R A' R^T, where A' is diagonal and R orthonormal
		// Use a fixed sequence of operations instead of looking at the largest element
		for (int i = 0; i < nr_sweeps; i++)
		{
			QuaternionJacobiRotate<Scalar, 0, 1>(A, Q);
			QuaternionJacobiRotate<Scalar, 0, 2>(A, Q);
			QuaternionJacobiRotate<Scalar, 1, 2>(A, Q);
		}

		// Normalize the rotation quaternion
		Q.normalize();

		return nr_sweeps * 3;
	}
	
	template<typename Scalar>
	int SelfAdjointJacobiEigenQuatIncrementalSweeps(Eigen::Matrix<Scalar, 3, 3>& A, Eigen::Matrix<Scalar, 3, 3>& R)
	{
		Eigen::Quaternion<Scalar> Q;
		int iter = SelfAdjointJacobiEigenQuatIncrementalSweeps(A, Q);

		R = Q.toRotationMatrix();

		return iter;
	}

	template<typename Scalar>
	int SelfAdjointJacobiEigenQuatSweeps(Eigen::Matrix<Scalar, 3, 3>& A, Eigen::Matrix<Scalar, 3, 3>& R)
	{
		Eigen::Quaternion<Scalar> Q;
		int iter = SelfAdjointJacobiEigenQuatSweeps(A, Q);

		R = Q.toRotationMatrix();
		A = R.transpose() * A * R;

		return iter;
	}
}}
