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

namespace Vcl { namespace Mathematics
{
	VCL_STRONG_INLINE float sq(float a)
	{
		return a*a;
	}

	VCL_STRONG_INLINE double rsqrt(double in)
	{
	   return 1.0 / ::sqrt(in);
	}


	VCL_STRONG_INLINE double rcp(double in)
	{
		return 1.0 / in;
	}

	VCL_STRONG_INLINE double sgn(double x)
	{
		return (double) ((0.0 < x) - (x < 0.0));
	}

	VCL_STRONG_INLINE double abs(double a)
	{
		return ::fabs(a);
	}

	VCL_STRONG_INLINE double sqrt(double a)
	{
		return ::sqrt(a);
	}

	VCL_STRONG_INLINE double max(double a, double b)
	{
#ifdef VCL_VECTORIZE_SSE
		double z;
		_mm_store_sd(&z, _mm_max_sd(_mm_set_sd(a), _mm_set_sd(b)));
		return z;
#else
		return std::max(a, b);
#endif
	}

	VCL_STRONG_INLINE double min(double a, double b)
	{
#ifdef VCL_VECTORIZE_SSE
		double z;
		_mm_store_sd(&z, _mm_min_sd(_mm_set_sd(a), _mm_set_sd(b)));
		return z;
#else
		return std::min(a, b);
#endif
	}

	VCL_STRONG_INLINE Eigen::Matrix<double, 12, 1> mul(const Eigen::Matrix<double, 12, 12>& A, const Eigen::Matrix<double, 12, 1>& x)
	{
		static_assert((Eigen::Matrix<double, 12, 12>::Options & Eigen::RowMajorBit) == 0, "Only column major matrices are supported.");

		// Inspecting the code shows that it is compiled using SSE
		Eigen::Matrix<double, 12, 1> acc = A.col(0) * x(0);
		for (int i = 1; i < 12; i++)
		{
			acc += A.col(i) * x(i);
		}

		return acc;
	}

	VCL_STRONG_INLINE Eigen::Matrix<double, 12, 1> mul(const Eigen::Matrix<double, 12, 12>& A, Eigen::Matrix<double, 12, 1>&& x)
	{
		static_assert((Eigen::Matrix<double, 12, 12>::Options & Eigen::RowMajorBit) == 0, "Only column major matrices are supported.");

		// Inspecting the code shows that it is compiled using SSE
		Eigen::Matrix<double, 12, 1> acc = A.col(0) * x(0);
		for (int i = 1; i < 12; i++)
		{
			acc += A.col(i) * x(i);
		}

		return acc;
	}

	template <typename T>
	VCL_STRONG_INLINE int sgn(T x, std::false_type is_signed)
	{
		return T(0) < x;
	}

	template <typename T>
	VCL_STRONG_INLINE int sgn(T x, std::true_type is_signed)
	{
		return (T(0) < x) - (x < T(0));
	}

	template <typename T>
	VCL_STRONG_INLINE int sgn(T x)
	{
		return sgn(x, std::is_signed<T>());
	}

	VCL_STRONG_INLINE float sgn(float x)
	{
		return (float) ((0.0f < x) - (x < 0.0f));
	}

	VCL_STRONG_INLINE float abs(float a)
	{
		return ::fabs(a);
	}

	VCL_STRONG_INLINE float sqrt(float a)
	{
		return ::sqrt(a);
	}

	VCL_STRONG_INLINE float rsqrt(float in)
	{
#ifdef VCL_VECTORIZE_SSE
		float out;

		const __m128 v = _mm_set_ss(in);
		const __m128 nr = _mm_rsqrt_ss(v);
		const __m128 muls = _mm_mul_ss(_mm_mul_ss(nr, nr), v);
		const __m128 beta = _mm_mul_ss(_mm_set_ss(0.5f), nr);
		const __m128 gamma = _mm_sub_ss(_mm_set_ss(3.0f), muls);

		_mm_store_ss(&out, _mm_mul_ss(beta, gamma));
		return out;
#else
		return 1 / std::sqrt(in);
#endif
	}

	VCL_STRONG_INLINE float rcp(float f)
	{
#ifdef VCL_VECTORIZE_SSE
		float out;

		__m128 v = _mm_set_ss(f);

		__m128 nr = _mm_rcp_ss(v);
		__m128 muls = _mm_mul_ss(_mm_mul_ss(nr, nr), v);
		__m128 dbl = _mm_add_ss(nr, nr);

		// Filter out zero input to ensure 
		__m128 mask = _mm_cmpeq_ss(v, _mm_setzero_ps());
		__m128 filtered = _mm_andnot_ps(mask, muls);
		__m128 result = _mm_sub_ss(dbl, filtered);

		_mm_store_ss(&out, result);
		return out;
#else
		return 1 / f;
#endif
	}

	VCL_STRONG_INLINE float max(float a, float b)
	{
#ifdef VCL_VECTORIZE_SSE
		float z;
		_mm_store_ss(&z, _mm_max_ss(_mm_set_ss(a), _mm_set_ss(b)));
		return z;
#else
		return (a > b) ? a : b;
#endif
	}

	VCL_STRONG_INLINE float min(float a, float b)
	{
#ifdef VCL_VECTORIZE_SSE
		float z;
		_mm_store_ss(&z, _mm_min_ss(_mm_set_ss(a), _mm_set_ss(b)));
		return z;
#else
		return (a < b) ? a : b;
#endif
	}

	VCL_STRONG_INLINE Eigen::Matrix<float, 12, 1> mul(const Eigen::Matrix<float, 12, 12>& A, const Eigen::Matrix<float, 12, 1>& x)
	{
		static_assert((Eigen::Matrix<float, 12, 12>::Options & Eigen::RowMajorBit) == 0, "Only column major matrices are supported.");

		// Inspecting the code shows that it is compiled using SSE
		Eigen::Matrix<float, 12, 1> acc = A.col(0) * x(0);
		for (int i = 1; i < 12; i++)
		{
			acc += A.col(i) * x(i);
		}

		return acc;
	}

	VCL_STRONG_INLINE Eigen::Matrix<float, 12, 1> mul(const Eigen::Matrix<float, 12, 12>& A, Eigen::Matrix<float, 12, 1>&& x)
	{
		static_assert((Eigen::Matrix<float, 12, 12>::Options & Eigen::RowMajorBit) == 0, "Only column major matrices are supported.");

		// Inspecting the code shows that it is compiled using SSE
		Eigen::Matrix<float, 12, 1> acc = A.col(0) * x(0);
		for (int i = 1; i < 12; i++)
		{
			acc += A.col(i) * x(i);
		}

		return acc;
	}

	template<typename T>
	VCL_STRONG_INLINE T clamp(T&& v, T&& l, T&& u)
	{
		return min(max(v, l), u);
	}

	//Based on: http://realtimecollisiondetection.net/pubs/GDC08_Ericson_Physics_Tutorial_Numerical_Robustness.ppt
	VCL_STRONG_INLINE bool equal(double x, double y, double tol = 0.0)
	{
		return abs(x - y) <= tol * fmax(1.0, fmax(abs(x), abs(y)));
	}

	VCL_STRONG_INLINE bool equal(float x, float y, float tol = 0.0f)
	{
		return abs(x - y) <= tol * fmax(1.0f, fmax(abs(x), abs(y)));
	}

	template<typename Scalar, int Rows, int Cols>
	VCL_STRONG_INLINE bool equal
	(
		const Eigen::Matrix<Scalar, Rows, Cols>& x,
		const Eigen::Matrix<Scalar, Rows, Cols>& y,
		Scalar tol = 0
	)
	{
		bool eq = true;
		for (int c = 0; c < Cols; c++)
			for (int r = 0; r < Rows; r++)
				eq = eq && equal(x(r, c), y(r, c), tol);

		return eq;
	}

	template<typename T>
	constexpr T pi()
	{
		return (T) 3.14159265358979323846;
	}
}}
