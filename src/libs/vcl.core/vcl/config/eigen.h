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

// Configure vectorisation
#ifdef VCL_VECTORIZE_SSE3
#	define EIGEN_VECTORIZE_SSE3
#endif
#ifdef VCL_VECTORIZE_SSSE3
#	define EIGEN_VECTORIZE_SSE3
#	define EIGEN_VECTORIZE_SSSE3
#endif
#ifdef VCL_VECTORIZE_SSE4_1
#	define EIGEN_VECTORIZE_SSE3
#	define EIGEN_VECTORIZE_SSSE3
#	define EIGEN_VECTORIZE_SSE4_1
#endif
#ifdef VCL_VECTORIZE_SSE4_2
#	define EIGEN_VECTORIZE_SSE3
#	define EIGEN_VECTORIZE_SSSE3
#	define EIGEN_VECTORIZE_SSE4_1
#	define EIGEN_VECTORIZE_SSE4_2
#endif
#ifdef VCL_VECTORIZE_AVX
#	define EIGEN_VECTORIZE_SSE3
#	define EIGEN_VECTORIZE_SSSE3
#	define EIGEN_VECTORIZE_SSE4_1
#	define EIGEN_VECTORIZE_SSE4_2
#endif

VCL_BEGIN_EXTERNAL_HEADERS
// Eigen library
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

// C++ standard libary
#include <vector>
VCL_END_EXTERNAL_HEADERS

// Cwise ops
namespace Eigen
{
	template<typename Scalar>
	struct CwiseClampOp
	{
		CwiseClampOp(const Scalar& inf, const Scalar& sup) : m_inf(inf), m_sup(sup) {}
		const Scalar operator()(const Scalar& x) const { return x<m_inf ? m_inf : (x>m_sup ? m_sup : x); }
		Scalar m_inf, m_sup;
	};

	template<typename Scalar>
	struct CwiseThresholdOp
	{
		CwiseThresholdOp(const Scalar& inf) : mInf(inf) {}
		const Scalar operator()(const Scalar& x) const { return x < mInf ? 0 : x; }
		Scalar mInf;
	};
	
	template<typename Scalar>
	struct CwiseInverseWithThresholdOp
	{
		CwiseInverseWithThresholdOp(const Scalar& tol) : mTol(tol) {}
		const Scalar operator()(const Scalar& x) const { return x < mTol ? 0 : 1.0 / x; }
		Scalar mTol;
	};
	
	template<typename Scalar>
	struct CwiseFractionalPartOp
	{
		const Scalar operator()(const Scalar& x) const { Scalar intPart = 0; return std::modf(x, &intPart); }
	};
	
	template<typename Scalar>
	struct CwiseIntegralPartOp
	{
		const Scalar operator()(const Scalar& x) const { Scalar intPart = 0; std::modf(x, &intPart); return intPart; }
	};

	template<typename Scalar>
	struct CwiseFloorOp
	{
		const Scalar operator()(const Scalar& x) const { return std::floor(x); }
	};

	template<typename Scalar>
	struct CwiseCeilOp
	{
		const Scalar operator()(const Scalar& x) const { return std::ceil(x); }
	};
}

// Some additional Eigen typedefs
namespace Eigen
{
	typedef Eigen::AlignedBox<float, 3> AlignedBox3f;
	typedef Eigen::ParametrizedLine<float, 3> ParametrizedLine3f;
	typedef Eigen::Hyperplane<float, 3> Hyperplane3f;
	typedef Eigen::Quaternion<float> Quaternionf;

	typedef Eigen::AlignedBox<double, 3> AlignedBox3d;
	typedef Eigen::ParametrizedLine<double, 3> ParametrizedLine3d;
	typedef Eigen::Hyperplane<double, 3> Hyperplane3d;
	typedef Eigen::Quaternion<double> Quaterniond;
}

// Typedefs for VCL
namespace Vcl
{
	typedef Eigen::AlignedBox<float, 3> AlignedBox3f;

	typedef Eigen::Matrix<float, 2, 1> Vector2f;
	typedef Eigen::Matrix<float, 3, 1> Vector3f;
	typedef Eigen::Matrix<float, 4, 1> Vector4f;
	typedef Eigen::Matrix<float, 2, 2> Matrix2f;
	typedef Eigen::Matrix<float, 3, 3> Matrix3f;
	typedef Eigen::Matrix<float, 4, 4> Matrix4f;
	
	typedef Eigen::Matrix<float, 2, 1, Eigen::ColMajor|Eigen::Unaligned> UnalignedVector2f;
	typedef Eigen::Matrix<float, 3, 1, Eigen::ColMajor|Eigen::Unaligned> UnalignedVector3f;
	typedef Eigen::Matrix<float, 4, 1, Eigen::ColMajor|Eigen::Unaligned> UnalignedVector4f;

	typedef Eigen::Quaternion<float, Eigen::Unaligned> UnalignedQuaternionf;
}
