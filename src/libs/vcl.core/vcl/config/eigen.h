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
#ifndef __EMSCRIPTEN__
#	if defined VCL_VECTORIZE_AVX
#		define EIGEN_VECTORIZE_AVX
#		define EIGEN_VECTORIZE_SSE4_2
#		define EIGEN_VECTORIZE_SSE4_1
#		define EIGEN_VECTORIZE_SSSE3
#		define EIGEN_VECTORIZE_SSE3
#	elif defined VCL_VECTORIZE_SSE
#		if defined VCL_VECTORIZE_SSE4_2
#			define EIGEN_VECTORIZE_SSE4_2
#			define EIGEN_VECTORIZE_SSE4_1
#			define EIGEN_VECTORIZE_SSSE3
#			define EIGEN_VECTORIZE_SSE3
#		elif defined VCL_VECTORIZE_SSE4_1
#			define EIGEN_VECTORIZE_SSE4_1
#			define EIGEN_VECTORIZE_SSSE3
#			define EIGEN_VECTORIZE_SSE3
#		elif defined VCL_VECTORIZE_SSSE3
#			define EIGEN_VECTORIZE_SSSE3
#			define EIGEN_VECTORIZE_SSE3
#		elif defined VCL_VECTORIZE_SSE3
#			define EIGEN_VECTORIZE_SSE3
#		endif
#	endif
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
namespace Eigen {
	template<typename Scalar>
	struct CwiseClampOp
	{
		CwiseClampOp(const Scalar& inf, const Scalar& sup)
		: m_inf(inf), m_sup(sup) {}
		const Scalar operator()(const Scalar& x) const { return x < m_inf ? m_inf : (x > m_sup ? m_sup : x); }
		Scalar m_inf, m_sup;
	};

	template<typename Scalar>
	struct CwiseThresholdOp
	{
		CwiseThresholdOp(const Scalar& inf)
		: mInf(inf) {}
		const Scalar operator()(const Scalar& x) const { return x < mInf ? 0 : x; }
		Scalar mInf;
	};

	template<typename Scalar>
	struct CwiseInverseWithThresholdOp
	{
		CwiseInverseWithThresholdOp(const Scalar& tol)
		: mTol(tol) {}
		const Scalar operator()(const Scalar& x) const { return x < mTol ? 0 : 1.0 / x; }
		Scalar mTol;
	};

	template<typename Scalar>
	struct CwiseFractionalPartOp
	{
		const Scalar operator()(const Scalar& x) const
		{
			Scalar intPart = 0;
			return std::modf(x, &intPart);
		}
	};

	template<typename Scalar>
	struct CwiseIntegralPartOp
	{
		const Scalar operator()(const Scalar& x) const
		{
			Scalar intPart = 0;
			std::modf(x, &intPart);
			return intPart;
		}
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
namespace Eigen {
	using AlignedBox3f = Eigen::AlignedBox<float, 3>;
	using ParametrizedLine3f = Eigen::ParametrizedLine<float, 3>;
	using Hyperplane3f = Eigen::Hyperplane<float, 3>;
	using Quaternionf = Eigen::Quaternion<float>;

	using AlignedBox3d = Eigen::AlignedBox<double, 3>;
	using ParametrizedLine3d = Eigen::ParametrizedLine<double, 3>;
	using Hyperplane3d = Eigen::Hyperplane<double, 3>;
	using Quaterniond = Eigen::Quaternion<double>;

	using Vector2ui = Eigen::Matrix<unsigned int, 2, 1>;
	using Vector3ui = Eigen::Matrix<unsigned int, 3, 1>;
	using Vector4ui = Eigen::Matrix<unsigned int, 4, 1>;
}

// Typedefs for VCL
namespace Vcl {
	using AlignedBox3f = Eigen::AlignedBox<float, 3>;

	using Vector2f = Eigen::Matrix<float, 2, 1>;
	using Vector3f = Eigen::Matrix<float, 3, 1>;
	using Vector4f = Eigen::Matrix<float, 4, 1>;
	using Matrix2f = Eigen::Matrix<float, 2, 2>;
	using Matrix3f = Eigen::Matrix<float, 3, 3>;
	using Matrix4f = Eigen::Matrix<float, 4, 4>;

	using UnalignedVector2f = Eigen::Matrix<float, 2, 1, Eigen::ColMajor | Eigen::DontAlign>;
	using UnalignedVector3f = Eigen::Matrix<float, 3, 1, Eigen::ColMajor | Eigen::DontAlign>;
	using UnalignedVector4f = Eigen::Matrix<float, 4, 1, Eigen::ColMajor | Eigen::DontAlign>;

	using UnalignedQuaternionf = Eigen::Quaternion<float, Eigen::DontAlign>;

	using Vector2d = Eigen::Matrix<double, 2, 1>;
	using Vector3d = Eigen::Matrix<double, 3, 1>;
	using Vector4d = Eigen::Matrix<double, 4, 1>;
	using Matrix2d = Eigen::Matrix<double, 2, 2>;
	using Matrix3d = Eigen::Matrix<double, 3, 3>;
	using Matrix4d = Eigen::Matrix<double, 4, 4>;
}
