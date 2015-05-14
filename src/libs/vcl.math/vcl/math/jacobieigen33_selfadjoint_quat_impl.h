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
	VCL_STRONG_INLINE Eigen::Matrix<Real, 2, 1> ApproxJacobiRotationQuaternion(Real a11, Real a12, Real a22)
	{
		//const Real pi = Real(3.1415926535897932384626433832795);

		//const Real gamma = Real(3) + Real(2) * Real(std::sqrt(Real(2)));
		const Real gamma = Real(5.8284271247461900976033774484194);

		//const Real c_star = Real(std::cos(pi / Real(8)));
		const Real c_star = Real(0.92387953251128675612818318939679);
		
		//const Real s_star = Real(std::sin(pi / Real(8)));
		const Real s_star = Real(0.3826834323650897717284599840304);
		
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

	//template<typename Scalar, int p, int q>
	//void QuaternionJacobiRotateIncremental(Eigen::Matrix<Scalar, 3, 3>& m, Eigen::Matrix<Scalar, 3, 3>& R)
	//{
	//	using namespace Eigen;
	//
	//	// Rotates A through phi in pq-plane to set A(p, q) = 0.
	//	// Rotation stored in R whose columns are eigenvectors of A
	//
	//	// Use approximate rotations
	//	auto cs = ApproximateGivensRotationQuaternion(m(p, p), m(p, q), m(q, q));
	//	Scalar c = cs(0);
	//	Scalar s = cs(1);
	//
	//	if (p == 0 && q == 1)
	//	{
	//		Eigen::Quaternion<Scalar> qq(c, 0, 0, s);
	//		qq.normalize();
	//
	//		m = qq.toRotationMatrix().transpose() * m * qq.toRotationMatrix();
	//		R *= qq.toRotationMatrix();
	//
	//		return;
	//	}
	//	if (p == 0 && q == 2)
	//	{
	//		Eigen::Quaternion<Scalar> qq(c, 0, -s, 0);
	//		qq.normalize();
	//
	//		m = qq.toRotationMatrix().transpose() * m * qq.toRotationMatrix();
	//		R *= qq.toRotationMatrix();
	//
	//		return;
	//	}
	//	if (p == 1 && q == 2)
	//	{
	//		Eigen::Quaternion<Scalar> qq(c, s, 0, 0);
	//		qq.normalize();
	//
	//		m = qq.toRotationMatrix().transpose() * m * qq.toRotationMatrix();
	//		R *= qq.toRotationMatrix();
	//
	//		return;
	//	}
	//}
	
	template<typename Scalar, int p, int q>
	void QuaternionJacobiRotateIncremental(Eigen::Matrix<Scalar, 3, 3>& M, Eigen::Quaternion<Scalar>& Q)
	{
		// Rotates A through phi in pq-plane to set M(p, q) = 0.
		auto cs = ApproxJacobiRotationQuaternion(M(p, p), M(p, q), M(q, q));
		Scalar c = cs(0);
		Scalar s = cs(1);

		if (p == 0 && q == 1)
		{
			// Build the rotation matrix
			Eigen::Matrix<Scalar, 3, 3> R;
			R(0, 0) = c*c - s*s;
			R(1, 0) = Scalar(2)*s*c;
			R(2, 0) = 0;

			R(0, 1) = Scalar(-2)*s*c;
			R(1, 1) = c*c - s*s;
			R(2, 1) = 0;

			R(0, 2) = 0;
			R(1, 2) = 0;
			R(2, 2) = c*c + s*s;
			
			// Transform the matrices
			M = R.transpose() * M * R;
			Q *= Eigen::Quaternion<Scalar>(c, 0, 0, s);
		}
		else if (p == 0 && q == 2)
		{
			// Build the rotation matrix
			Eigen::Matrix<Scalar, 3, 3> R;
			R(0, 0) = c*c - s*s;
			R(1, 0) = 0;
			R(2, 0) = Scalar(-2)*s*c;

			R(0, 1) = 0;
			R(1, 1) = c*c + s*s;
			R(2, 1) = 0;

			R(0, 2) = Scalar(2)*s*c;
			R(1, 2) = 0;
			R(2, 2) = c*c - s*s;
			
			// Transform the matrices
			M = R.transpose() * M * R;
			Q *= Eigen::Quaternion<Scalar>(c, 0, -s, 0);
		}
		else if (p == 1 && q == 2)
		{
			// Build the rotation matrix
			Eigen::Matrix<Scalar, 3, 3> R;
			R(0, 0) = c*c + s*s;
			R(1, 0) = 0;
			R(2, 0) = 0;

			R(0, 1) = 0;
			R(1, 1) = c*c - s*s;
			R(2, 1) = Scalar(2)*s*c;

			R(0, 2) = 0;
			R(1, 2) = Scalar(-2)*s*c;
			R(2, 2) = c*c - s*s;
			
			// Transform the matrices
			M = R.transpose() * M * R;
			Q *= Eigen::Quaternion<Scalar>(c, s, 0, 0);
		}
	}

	template<typename Scalar, int p, int q>
	void QuaternionJacobiRotate(const Eigen::Matrix<Scalar, 3, 3>& M, Eigen::Quaternion<Scalar>& Q)
	{
		Require(all(equal(Q.norm(), Scalar(1), Scalar(1e-6))), "Quaternion is normalized.");
		
		Scalar x = Q.x();
		Scalar y = Q.y();
		Scalar z = Q.z();
		Scalar w = Q.w();
		
		if (p == 0 && q == 1)
		{
			// Maple generated code
			Scalar D00 = ((w * w + x * x - y * y - z * z) * M(0, 0) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 0) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 0)) * (w * w + x * x - y * y - z * z) + ((w * w + x * x - y * y - z * z) * M(0, 1) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 1) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 1)) * (Scalar(2) * x * y + Scalar(2) * w * z) + ((w * w + x * x - y * y - z * z) * M(0, 2) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 2) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 2)) * (Scalar(2) * x * z - Scalar(2) * w * y);
			Scalar D11 = ((Scalar(2) * x * y - Scalar(2) * w * z) * M(0, 0) + (w * w - x * x + y * y - z * z) * M(1, 0) + (Scalar(2) * y * z + Scalar(2) * w * x) * M(2, 0)) * (Scalar(2) * x * y - Scalar(2) * w * z) + ((Scalar(2) * x * y - Scalar(2) * w * z) * M(0, 1) + (w * w - x * x + y * y - z * z) * M(1, 1) + (Scalar(2) * y * z + Scalar(2) * w * x) * M(2, 1)) * (w * w - x * x + y * y - z * z) + ((Scalar(2) * x * y - Scalar(2) * w * z) * M(0, 2) + (w * w - x * x + y * y - z * z) * M(1, 2) + (Scalar(2) * y * z + Scalar(2) * w * x) * M(2, 2)) * (Scalar(2) * y * z + Scalar(2) * w * x);
			Scalar D01 = ((w * w + x * x - y * y - z * z) * M(0, 0) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 0) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 0)) * (Scalar(2) * x * y - Scalar(2) * w * z) + ((w * w + x * x - y * y - z * z) * M(0, 1) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 1) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 1)) * (w * w - x * x + y * y - z * z) + ((w * w + x * x - y * y - z * z) * M(0, 2) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 2) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 2)) * (Scalar(2) * y * z + Scalar(2) * w * x);
			
			// Rotates A through phi in pq-plane to set D(p, q) = 0.
			auto cs = ApproxJacobiRotationQuaternion(D00, D01, D11);
			Scalar c = cs(0);
			Scalar s = cs(1);

			Check(all(equal(Eigen::Quaternion<Scalar>(c, 0, 0, s).norm(), Scalar(1), Scalar(1e-6))), "Quaternion is normalized.");

			Q *= Eigen::Quaternion<Scalar>(c, 0, 0, s);
		}
		else if (p == 0 && q == 2)
		{
			// Maple generated code
			Scalar D00 = ((w * w + x * x - y * y - z * z) * M(0, 0) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 0) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 0)) * (w * w + x * x - y * y - z * z) + ((w * w + x * x - y * y - z * z) * M(0, 1) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 1) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 1)) * (Scalar(2) * x * y + Scalar(2) * w * z) + ((w * w + x * x - y * y - z * z) * M(0, 2) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 2) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 2)) * (Scalar(2) * x * z - Scalar(2) * w * y);
			Scalar D22 = ((Scalar(2) * x * z + Scalar(2) * w * y) * M(0, 0) + (Scalar(2) * y * z - Scalar(2) * w * x) * M(1, 0) + (w * w - x * x - y * y + z * z) * M(2, 0)) * (Scalar(2) * x * z + Scalar(2) * w * y) + ((Scalar(2) * x * z + Scalar(2) * w * y) * M(0, 1) + (Scalar(2) * y * z - Scalar(2) * w * x) * M(1, 1) + (w * w - x * x - y * y + z * z) * M(2, 1)) * (Scalar(2) * y * z - Scalar(2) * w * x) + ((Scalar(2) * x * z + Scalar(2) * w * y) * M(0, 2) + (Scalar(2) * y * z - Scalar(2) * w * x) * M(1, 2) + (w * w - x * x - y * y + z * z) * M(2, 2)) * (w * w - x * x - y * y + z * z);
			Scalar D02 = ((w * w + x * x - y * y - z * z) * M(0, 0) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 0) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 0)) * (Scalar(2) * x * z + Scalar(2) * w * y) + ((w * w + x * x - y * y - z * z) * M(0, 1) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 1) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 1)) * (Scalar(2) * y * z - Scalar(2) * w * x) + ((w * w + x * x - y * y - z * z) * M(0, 2) + (Scalar(2) * x * y + Scalar(2) * w * z) * M(1, 2) + (Scalar(2) * x * z - Scalar(2) * w * y) * M(2, 2)) * (w * w - x * x - y * y + z * z);
			
			// Rotates A through phi in pq-plane to set D(p, q) = 0.
			auto cs = ApproxJacobiRotationQuaternion(D00, D02, D22);
			Scalar c = cs(0);
			Scalar s = cs(1);
			
			Check(all(equal(Eigen::Quaternion<Scalar>(c, 0, -s, 0).norm(), Scalar(1), Scalar(1e-6))), "Quaternion is normalized.");

			Q *= Eigen::Quaternion<Scalar>(c, 0, -s, 0);
		}
		else if (p == 1 && q == 2)
		{
			// Maple generated code
			Scalar D11 = ((Scalar(2) * x * y - Scalar(2) * w * z) * M(0, 0) + (w * w - x * x + y * y - z * z) * M(1, 0) + (Scalar(2) * y * z + Scalar(2) * w * x) * M(2, 0)) * (Scalar(2) * x * y - Scalar(2) * w * z) + ((Scalar(2) * x * y - Scalar(2) * w * z) * M(0, 1) + (w * w - x * x + y * y - z * z) * M(1, 1) + (Scalar(2) * y * z + Scalar(2) * w * x) * M(2, 1)) * (w * w - x * x + y * y - z * z) + ((Scalar(2) * x * y - Scalar(2) * w * z) * M(0, 2) + (w * w - x * x + y * y - z * z) * M(1, 2) + (Scalar(2) * y * z + Scalar(2) * w * x) * M(2, 2)) * (Scalar(2) * y * z + Scalar(2) * w * x);
			Scalar D22 = ((Scalar(2) * x * z + Scalar(2) * w * y) * M(0, 0) + (Scalar(2) * y * z - Scalar(2) * w * x) * M(1, 0) + (w * w - x * x - y * y + z * z) * M(2, 0)) * (Scalar(2) * x * z + Scalar(2) * w * y) + ((Scalar(2) * x * z + Scalar(2) * w * y) * M(0, 1) + (Scalar(2) * y * z - Scalar(2) * w * x) * M(1, 1) + (w * w - x * x - y * y + z * z) * M(2, 1)) * (Scalar(2) * y * z - Scalar(2) * w * x) + ((Scalar(2) * x * z + Scalar(2) * w * y) * M(0, 2) + (Scalar(2) * y * z - Scalar(2) * w * x) * M(1, 2) + (w * w - x * x - y * y + z * z) * M(2, 2)) * (w * w - x * x - y * y + z * z);
			Scalar D12 = ((Scalar(2) * x * y - Scalar(2) * w * z) * M(0, 0) + (w * w - x * x + y * y - z * z) * M(1, 0) + (Scalar(2) * y * z + Scalar(2) * w * x) * M(2, 0)) * (Scalar(2) * x * z + Scalar(2) * w * y) + ((Scalar(2) * x * y - Scalar(2) * w * z) * M(0, 1) + (w * w - x * x + y * y - z * z) * M(1, 1) + (Scalar(2) * y * z + Scalar(2) * w * x) * M(2, 1)) * (Scalar(2) * y * z - Scalar(2) * w * x) + ((Scalar(2) * x * y - Scalar(2) * w * z) * M(0, 2) + (w * w - x * x + y * y - z * z) * M(1, 2) + (Scalar(2) * y * z + Scalar(2) * w * x) * M(2, 2)) * (w * w - x * x - y * y + z * z);

			// Rotates A through phi in pq-plane to set D(p, q) = 0.
			auto cs = ApproxJacobiRotationQuaternion(D11, D12, D22);
			Scalar c = cs(0);
			Scalar s = cs(1);
			
			Check(all(equal(Eigen::Quaternion<Scalar>(c, s, 0, 0).norm(), Scalar(1), Scalar(1e-6))), "Quaternion is normalized.");

			Q *= Eigen::Quaternion<Scalar>(c, s, 0, 0);
		}
	}
	
	template<typename Scalar>
	void QuaternionJacobiRotate(const Eigen::Matrix<Scalar, 3, 3>& M, int p, int q, Eigen::Quaternion<Scalar>& Q)
	{
		Require(equal(Q.norm(), 1, 1e-6), "Quaternion is normalized.");
		
		// Gerneric code
		//Eigen::Matrix<Scalar, 3, 3> R = Q.toRotationMatrix();
		//Eigen::Matrix<Scalar, 3, 3> D = R.transpose() * M * R;
		
		Eigen::Matrix<Scalar, 3, 3> D;
		Scalar x = Q.x();
		Scalar y = Q.y();
		Scalar z = Q.z();
		Scalar w = Q.w();
		
		// Maple generated code
		D(0, 0) = ((w * w + x * x - y * y - z * z) * M(0, 0) + (2 * x * y + 2 * w * z) * M(1, 0) + (2 * x * z - 2 * w * y) * M(2, 0)) * (w * w + x * x - y * y - z * z) + ((w * w + x * x - y * y - z * z) * M(0, 1) + (2 * x * y + 2 * w * z) * M(1, 1) + (2 * x * z - 2 * w * y) * M(2, 1)) * (2 * x * y + 2 * w * z) + ((w * w + x * x - y * y - z * z) * M(0, 2) + (2 * x * y + 2 * w * z) * M(1, 2) + (2 * x * z - 2 * w * y) * M(2, 2)) * (2 * x * z - 2 * w * y);
		D(1, 1) = ((2 * x * y - 2 * w * z) * M(0, 0) + (w * w - x * x + y * y - z * z) * M(1, 0) + (2 * y * z + 2 * w * x) * M(2, 0)) * (2 * x * y - 2 * w * z) + ((2 * x * y - 2 * w * z) * M(0, 1) + (w * w - x * x + y * y - z * z) * M(1, 1) + (2 * y * z + 2 * w * x) * M(2, 1)) * (w * w - x * x + y * y - z * z) + ((2 * x * y - 2 * w * z) * M(0, 2) + (w * w - x * x + y * y - z * z) * M(1, 2) + (2 * y * z + 2 * w * x) * M(2, 2)) * (2 * y * z + 2 * w * x);
		D(2, 2) = ((2 * x * z + 2 * w * y) * M(0, 0) + (2 * y * z - 2 * w * x) * M(1, 0) + (w * w - x * x - y * y + z * z) * M(2, 0)) * (2 * x * z + 2 * w * y) + ((2 * x * z + 2 * w * y) * M(0, 1) + (2 * y * z - 2 * w * x) * M(1, 1) + (w * w - x * x - y * y + z * z) * M(2, 1)) * (2 * y * z - 2 * w * x) + ((2 * x * z + 2 * w * y) * M(0, 2) + (2 * y * z - 2 * w * x) * M(1, 2) + (w * w - x * x - y * y + z * z) * M(2, 2)) * (w * w - x * x - y * y + z * z);
		
		D(0, 1) = ((w * w + x * x - y * y - z * z) * M(0, 0) + (2 * x * y + 2 * w * z) * M(1, 0) + (2 * x * z - 2 * w * y) * M(2, 0)) * (2 * x * y - 2 * w * z) + ((w * w + x * x - y * y - z * z) * M(0, 1) + (2 * x * y + 2 * w * z) * M(1, 1) + (2 * x * z - 2 * w * y) * M(2, 1)) * (w * w - x * x + y * y - z * z) + ((w * w + x * x - y * y - z * z) * M(0, 2) + (2 * x * y + 2 * w * z) * M(1, 2) + (2 * x * z - 2 * w * y) * M(2, 2)) * (2 * y * z + 2 * w * x);
		D(0, 2) = ((w * w + x * x - y * y - z * z) * M(0, 0) + (2 * x * y + 2 * w * z) * M(1, 0) + (2 * x * z - 2 * w * y) * M(2, 0)) * (2 * x * z + 2 * w * y) + ((w * w + x * x - y * y - z * z) * M(0, 1) + (2 * x * y + 2 * w * z) * M(1, 1) + (2 * x * z - 2 * w * y) * M(2, 1)) * (2 * y * z - 2 * w * x) + ((w * w + x * x - y * y - z * z) * M(0, 2) + (2 * x * y + 2 * w * z) * M(1, 2) + (2 * x * z - 2 * w * y) * M(2, 2)) * (w * w - x * x - y * y + z * z);
		D(1, 2) = ((2 * x * y - 2 * w * z) * M(0, 0) + (w * w - x * x + y * y - z * z) * M(1, 0) + (2 * y * z + 2 * w * x) * M(2, 0)) * (2 * x * z + 2 * w * y) + ((2 * x * y - 2 * w * z) * M(0, 1) + (w * w - x * x + y * y - z * z) * M(1, 1) + (2 * y * z + 2 * w * x) * M(2, 1)) * (2 * y * z - 2 * w * x) + ((2 * x * y - 2 * w * z) * M(0, 2) + (w * w - x * x + y * y - z * z) * M(1, 2) + (2 * y * z + 2 * w * x) * M(2, 2)) * (w * w - x * x - y * y + z * z);

		// Rotates A through phi in pq-plane to set D(p, q) = 0.
		auto cs = ApproxJacobiRotationQuaternion(D(p, p), D(p, q), D(q, q));
		Scalar c = cs(0);
		Scalar s = cs(1);

		Eigen::Matrix<Scalar, 3, 1> v = Eigen::Matrix<Scalar, 3, 1>::Zero();
		int idx = 2 - (p + q - 1);
		v(idx) = (idx%2 == 0) ? s : -s;
		
		Q *= Eigen::Quaternion<Scalar>(c, v(0), v(1), v(2));

		Ensure(epsEqual(Eigen::Quaternion<Scalar>(c, v(0), v(1), v(2)).norm(), 1, 1e-6), "Quaternion is normalized.");
	}
	
	/*
	 *	Method based on the technical report:
	 *		2011 - McAdams, Selle, Tamstorf, Teran, Sifakis - Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
	 *	which is an extensive description of the method presented in
	 *		SIGGRAPH - 2011 - McAdams, Zhu, Selle, Empey, Tamstorf, Teran, Sifakis - Efficient elasticity for character skinning with contact and collisions
	 */
	template<typename Scalar>
	int SelfAdjointJacobiEigenQuatSweeps(Eigen::Matrix<Scalar, 3, 3>& A, Eigen::Quaternion<Scalar>& Q, int nr_sweeps = 5)
	{
		using namespace Eigen;
		
		// Initialize Q
		Q = Eigen::Quaternion<Scalar>::Identity();

		// Only for symmetric matrices!
		// A = R A' R^T, where A' is diagonal and R orthonormal
		// Use a fixed sequence of operations instead of looking at the largest element
		int iter = 0;
		for (int i = 0; i < nr_sweeps; i++)
		{
			QuaternionJacobiRotate<Scalar, 0, 1>(A, Q);
			QuaternionJacobiRotate<Scalar, 0, 2>(A, Q);
			QuaternionJacobiRotate<Scalar, 1, 2>(A, Q);

			iter += 3;
		}

		// Normalize the rotation quaternion
		Q.normalize();

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
