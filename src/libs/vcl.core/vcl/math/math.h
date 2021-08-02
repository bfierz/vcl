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

namespace Vcl { namespace Mathematics {
	VCL_STRONG_INLINE VCL_CPP_CONSTEXPR_11 float sq(float a) noexcept
	{
		return a * a;
	}

	VCL_STRONG_INLINE double rsqrt(double in) noexcept
	{
		return 1.0 / ::sqrt(in);
	}

	VCL_STRONG_INLINE VCL_CPP_CONSTEXPR_11 double rcp(double in) noexcept
	{
		return 1.0 / in;
	}

	VCL_STRONG_INLINE VCL_CPP_CONSTEXPR_11 double sgn(double x) noexcept
	{
		return static_cast<double>(static_cast<int>(0.0 < x) - static_cast<int>(x < 0.0));
	}

	VCL_STRONG_INLINE double abs(double a) noexcept
	{
		return ::fabs(a);
	}

	VCL_STRONG_INLINE double sqrt(double a) noexcept
	{
		return ::sqrt(a);
	}

	VCL_STRONG_INLINE double max(double a, double b) noexcept
	{
#ifdef VCL_VECTORIZE_SSE
		double z;
		_mm_store_sd(&z, _mm_max_sd(_mm_set_sd(a), _mm_set_sd(b)));
		return z;
#else
		return std::max(a, b);
#endif
	}

	VCL_STRONG_INLINE double min(double a, double b) noexcept
	{
#ifdef VCL_VECTORIZE_SSE
		double z;
		_mm_store_sd(&z, _mm_min_sd(_mm_set_sd(a), _mm_set_sd(b)));
		return z;
#else
		return std::min(a, b);
#endif
	}

	template<typename T>
	VCL_STRONG_INLINE VCL_CPP_CONSTEXPR_11 int sgn(T x, std::false_type) noexcept
	{
		return T(0) < x ? 1 : 0;
	}

	template<typename T>
	VCL_STRONG_INLINE VCL_CPP_CONSTEXPR_11 int sgn(T x, std::true_type) noexcept
	{
		return (T(0) < x ? 1 : 0) - (x < T(0) ? 1 : 0);
	}

	template<typename T>
	VCL_STRONG_INLINE VCL_CPP_CONSTEXPR_11 int sgn(T x) noexcept
	{
		return sgn(x, std::is_signed<T>());
	}

	VCL_STRONG_INLINE VCL_CPP_CONSTEXPR_11 float sgn(float x) noexcept
	{
		return static_cast<float>((0.0f < x) - (x < 0.0f));
	}

	VCL_STRONG_INLINE float abs(float a) noexcept
	{
		return ::fabs(a);
	}

	VCL_STRONG_INLINE float sqrt(float a) noexcept
	{
		return ::sqrt(a);
	}

	VCL_STRONG_INLINE float rsqrt(float in) noexcept
	{
#ifdef VCL_VECTORIZE_SSE
		const __m128 v = _mm_set_ss(in);
		const __m128 nr = _mm_rsqrt_ss(v);
		const __m128 muls = _mm_mul_ss(_mm_mul_ss(nr, nr), v);
		const __m128 beta = _mm_mul_ss(_mm_set_ss(0.5f), nr);
		const __m128 gamma = _mm_sub_ss(_mm_set_ss(3.0f), muls);

		float out{ 0.0f };
		_mm_store_ss(&out, _mm_mul_ss(beta, gamma));
		return out;
#else
		return 1 / std::sqrt(in);
#endif
	}

	VCL_STRONG_INLINE float rcp(float f) noexcept
	{
#ifdef VCL_VECTORIZE_SSE
		const __m128 v = _mm_set_ss(f);
		const __m128 nr = _mm_rcp_ss(v);
		const __m128 muls = _mm_mul_ss(_mm_mul_ss(nr, nr), v);
		const __m128 dbl = _mm_add_ss(nr, nr);

		// Filter out zero input to ensure
		const __m128 mask = _mm_cmpeq_ss(v, _mm_setzero_ps());
		const __m128 filtered = _mm_andnot_ps(mask, muls);
		const __m128 result = _mm_sub_ss(dbl, filtered);

		float out{ 0.0f };
		_mm_store_ss(&out, result);
		return out;
#else
		return 1 / f;
#endif
	}

	VCL_STRONG_INLINE float max(float a, float b) noexcept
	{
#ifdef VCL_VECTORIZE_SSE
		return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(a), _mm_set_ss(b)));
#else
		return (a > b) ? a : b;
#endif
	}

	VCL_STRONG_INLINE float min(float a, float b) noexcept
	{
#ifdef VCL_VECTORIZE_SSE
		return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(a), _mm_set_ss(b)));
#else
		return (a < b) ? a : b;
#endif
	}

	template<typename T>
	VCL_STRONG_INLINE T clamp(T&& v, T&& l, T&& u) noexcept
	{
		return min(max(v, l), u);
	}

	//Based on: http://realtimecollisiondetection.net/pubs/GDC08_Ericson_Physics_Tutorial_Numerical_Robustness.ppt
	VCL_STRONG_INLINE bool equal(double x, double y, double tol = 0.0) noexcept
	{
		return abs(x - y) <= tol * fmax(1.0, fmax(abs(x), abs(y)));
	}

	VCL_STRONG_INLINE bool equal(float x, float y, float tol = 0.0f) noexcept
	{
		return abs(x - y) <= tol * fmax(1.0f, fmax(abs(x), abs(y)));
	}

	template<typename Scalar, int Rows, int Cols>
	VCL_STRONG_INLINE bool equal
	(
		const Eigen::Matrix<Scalar, Rows, Cols>& x,
		const Eigen::Matrix<Scalar, Rows, Cols>& y,
		Scalar tol = 0
	) noexcept
	{
		bool eq = true;
		for (int c = 0; c < Cols; c++)
			for (int r = 0; r < Rows; r++)
				eq = eq && equal(x(r, c), y(r, c), tol);

		return eq;
	}

	template<typename T>
	VCL_CPP_CONSTEXPR_11 T pi()
	{
		return static_cast<T>(3.14159265358979323846);
	}

	template<typename T>
	VCL_CPP_CONSTEXPR_11 T rad2deg()
	{
		return static_cast<T>(180) / pi<T>();
	}

	template<typename T>
	VCL_CPP_CONSTEXPR_11 T deg2rad()
	{
		return pi<T>() / static_cast<T>(180);
	}
}}
