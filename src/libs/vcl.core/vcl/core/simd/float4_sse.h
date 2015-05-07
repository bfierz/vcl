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

// C++ Standard Library
#include <array>

// VCL 
#include <vcl/core/simd/bool4_sse.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 4>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(const VectorScalar<float, 4>& rhs)
		{
			set(rhs.get(0));
		}
		VCL_STRONG_INLINE VectorScalar(float s)
		{
			set(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar(float s0, float s1, float s2, float s3)
		{
			set(s0, s1, s2, s3);
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m128 F4)
		{
			set(F4);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator= (const VectorScalar<float, 4>& rhs) { set(rhs.get(0)); return *this; }

	public:
		VCL_STRONG_INLINE float operator[] (int idx) const
		{
			Require(0 <= idx && idx < 4, "Access is in range.");

			return _mmVCL_extract_ps(get(0), idx);
		}

		VCL_STRONG_INLINE __m128 get(int i = 0) const
		{
			Require(0 == i, "Access is in range.");

			return _data[i];
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> operator- () const
		{
			return (*this) * VectorScalar<float, 4>(-1);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> operator+ (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_add_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> operator- (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_sub_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> operator* (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_mul_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> operator/ (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_div_ps(get(0), rhs.get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator += (const VectorScalar<float, 4>& rhs)
		{
			set(_mm_add_ps(get(0), rhs.get(0)));
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator -= (const VectorScalar<float, 4>& rhs)
		{
			set(_mm_sub_ps(get(0), rhs.get(0)));
			return *this;
		}

		VCL_STRONG_INLINE VectorScalar<float, 4>& operator *= (const VectorScalar<float, 4>& rhs)
		{
			set(_mm_mul_ps(get(0), rhs.get(0)));
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator /= (const VectorScalar<float, 4>& rhs)
		{
			set(_mm_div_ps(get(0), rhs.get(0)));
			return *this;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator== (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmpeq_ps (get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator!= (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmpneq_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator<  (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmplt_ps (get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator<= (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmple_ps (get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator>  (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmpgt_ps (get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator>= (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmpge_ps (get(0), rhs.get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> abs()   const { return VectorScalar<float, 4>(_mm_abs_ps     (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> sin()   const { return VectorScalar<float, 4>(_mm_sin_ps     (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> cos()   const { return VectorScalar<float, 4>(_mm_cos_ps     (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> exp()   const { return VectorScalar<float, 4>(_mm_exp_ps     (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> log()   const { return VectorScalar<float, 4>(_mm_log_ps     (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> sgn()   const { return VectorScalar<float, 4>(_mm_sgn_ps     (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> sqrt()  const { return VectorScalar<float, 4>(_mm_sqrt_ps    (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> rcp()   const { return VectorScalar<float, 4>(_mmVCL_rcp_ps  (get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> rsqrt() const { return VectorScalar<float, 4>(_mmVCL_rsqrt_ps(get(0))); }

		VCL_STRONG_INLINE VectorScalar<float, 4> acos() const { return VectorScalar<float, 4>(_mm_acos_ps(get(0))); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> min(const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_min_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<float, 4> max(const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_max_ps(get(0), rhs.get(0))); }

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 4>& rhs);
		friend VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b);
		
	private:
		VCL_STRONG_INLINE void set(float s0)
		{
			_data[0] = _mm_set1_ps(s0);
		}
		VCL_STRONG_INLINE void set(float s0, float s1, float s2, float s3)
		{
			_data[0] = _mm_set_ps(s3, s2, s1, s0);
		}
		VCL_STRONG_INLINE void set(__m128 vec)
		{
			_data[0] = vec;
		}

	private:
		__m128 _data[1];
	};

	VCL_STRONG_INLINE VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b)
	{
#ifdef VCL_VECTORIZE_SSE4_1
		// SSE way
		return VectorScalar<float, 4>(_mm_blendv_ps(b.get(0), a.get(0), mask.mF4));
#else
		// Straight forward method
		// (b & ~mask) | (a & mask)
		return VectorScalar<float, 4>(_mm_or_ps(_mm_andnot_ps(mask.mF4, b.get(0)), _mm_and_ps(mask.mF4, a.get(0))));

		// xor-method
		// (((b ^ a) & mask)^b)
		//return VectorScalar<float, 4>(_mm_xor_ps(b.get(0), _mm_and_ps(mask.mF4, _mm_xor_ps(b.get(0), a.get(0)))));
#endif
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 4>& rhs)
	{
		VCL_ALIGN(16) float vars[4];
		_mm_store_ps(vars + 0, rhs.get(0));
		
		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3] << "'";
		return s;
	}
}
