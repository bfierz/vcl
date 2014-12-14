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
		VCL_STRONG_INLINE VectorScalar(float s)
		{
			mF4 = _mm_set1_ps(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar(float s0, float s1, float s2, float s3)
		{
			mF4 = _mm_set_ps(s3, s2, s1, s0);
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m128 F4) : mF4(F4) {}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator= (const VectorScalar<float, 4>& rhs) { mF4 = rhs.mF4; return *this; }

	public:
		VCL_STRONG_INLINE float operator[] (int idx) const
		{
			Require(0 <= idx && idx < 4, "Access is in range.");

			return _mmVCL_extract_ps(mF4, idx);
		}

		VCL_STRONG_INLINE explicit operator __m128() const
		{
			return mF4;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> operator- () const
		{
			return (*this) * VectorScalar<float, 4>(-1);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> operator+ (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_add_ps(mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> operator- (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_sub_ps(mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> operator* (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_mul_ps(mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> operator/ (const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_div_ps(mF4, rhs.mF4)); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator += (const VectorScalar<float, 4>& rhs)
		{
			mF4 = _mm_add_ps(mF4, rhs.mF4);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator -= (const VectorScalar<float, 4>& rhs)
		{
			mF4 = _mm_sub_ps(mF4, rhs.mF4);
			return *this;
		}

		VCL_STRONG_INLINE VectorScalar<float, 4>& operator *= (const VectorScalar<float, 4>& rhs)
		{
			mF4 = _mm_mul_ps(mF4, rhs.mF4);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 4>& operator /= (const VectorScalar<float, 4>& rhs)
		{
			mF4 = _mm_div_ps(mF4, rhs.mF4);
			return *this;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator== (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmpeq_ps (mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator!= (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmpneq_ps(mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator<  (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmplt_ps (mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator<= (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmple_ps (mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator>  (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmpgt_ps (mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator>= (const VectorScalar<float, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_cmpge_ps (mF4, rhs.mF4)); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> abs()  const { return VectorScalar<float, 4>(_mm_abs_ps (mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> sin()  const { return VectorScalar<float, 4>(_mm_sin_ps (mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> cos()  const { return VectorScalar<float, 4>(_mm_cos_ps (mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> exp()  const { return VectorScalar<float, 4>(_mm_exp_ps (mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> log()  const { return VectorScalar<float, 4>(_mm_log_ps (mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> sgn()  const { return VectorScalar<float, 4>(_mm_sgn_ps (mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> sqrt() const { return VectorScalar<float, 4>(_mm_sqrt_ps(mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> rcp()  const { return VectorScalar<float, 4>(_mm_rcp_ps (mF4)); }

		VCL_STRONG_INLINE VectorScalar<float, 4> acos() const { return VectorScalar<float, 4>(_mm_acos_ps(mF4)); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 4> min(const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_min_ps(mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<float, 4> max(const VectorScalar<float, 4>& rhs) const { return VectorScalar<float, 4>(_mm_max_ps(mF4, rhs.mF4)); }

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 4>& rhs);
		friend VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b);

	private:
		__m128 mF4;
	};

	VCL_STRONG_INLINE VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b)
	{
		// Straight forward method
		// (b & mask) | (a & ~mask)

		// Optimized method
		// (((b ^ a) & mask)^b)
		return VectorScalar<float, 4>(_mm_xor_ps(b.mF4, _mm_and_ps(mask.mF4, _mm_xor_ps(b.mF4, a.mF4))));
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 4>& rhs)
	{
		float VCL_ALIGN(16) vars[4];
		_mm_store_ps(vars + 0, rhs.mF4);
		
		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3] << "'";
		return s;
	}
}
