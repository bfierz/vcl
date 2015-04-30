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
#include <vcl/core/simd/bool16_avx.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_avx.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 16>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(float s)
		{
			mF8[0] = _mm256_set1_ps(s);
			mF8[1] = _mm256_set1_ps(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar
		(
			float s00, float s01, float s02, float s03, float s04, float s05, float s06, float s07,
			float s08, float s09, float s10, float s11, float s12, float s13, float s14, float s15
		)
		{
			mF8[0] = _mm256_set_ps(s07, s06, s05, s04, s03, s02, s01, s00);
			mF8[1] = _mm256_set_ps(s15, s14, s13, s12, s11, s10, s09, s08);
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m256 F4_0, __m256 F4_1)
		{
			mF8[0] = F4_0;
			mF8[1] = F4_1;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator = (const VectorScalar<float, 16>& rhs)
		{
			mF8[0] = rhs.mF8[0];
			mF8[1] = rhs.mF8[1];

			return *this;
		}

	public:
		VCL_STRONG_INLINE float operator[] (int idx) const
		{
			Require(0 <= idx && idx < 16, "Access is in range.");

			return _mmVCL_extract_ps(mF8[idx / 8], idx % 8);
		}

		VCL_STRONG_INLINE __m256 get(int i) const
		{
			Require(0 <= i && i < 2, "Access is in range.");

			return mF8[i];
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> operator- () const
		{
			return (*this) * VectorScalar<float, 16>(-1);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> operator+ (const VectorScalar<float, 16>& rhs) const { return VectorScalar<float, 16>(_mm256_add_ps(mF8[0], rhs.mF8[0]), _mm256_add_ps(mF8[1], rhs.mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> operator- (const VectorScalar<float, 16>& rhs) const { return VectorScalar<float, 16>(_mm256_sub_ps(mF8[0], rhs.mF8[0]), _mm256_sub_ps(mF8[1], rhs.mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> operator* (const VectorScalar<float, 16>& rhs) const { return VectorScalar<float, 16>(_mm256_mul_ps(mF8[0], rhs.mF8[0]), _mm256_mul_ps(mF8[1], rhs.mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> operator/ (const VectorScalar<float, 16>& rhs) const { return VectorScalar<float, 16>(_mm256_div_ps(mF8[0], rhs.mF8[0]), _mm256_div_ps(mF8[1], rhs.mF8[1])); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator += (const VectorScalar<float, 16>& rhs)
		{
			mF8[0] = _mm256_add_ps(mF8[0], rhs.mF8[0]);
			mF8[1] = _mm256_add_ps(mF8[1], rhs.mF8[1]);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator -= (const VectorScalar<float, 16>& rhs)
		{
			mF8[0] = _mm256_sub_ps(mF8[0], rhs.mF8[0]);
			mF8[1] = _mm256_sub_ps(mF8[1], rhs.mF8[1]);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator *= (const VectorScalar<float, 16>& rhs)
		{
			mF8[0] = _mm256_mul_ps(mF8[0], rhs.mF8[0]);
			mF8[1] = _mm256_mul_ps(mF8[1], rhs.mF8[1]);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator /= (const VectorScalar<float, 16>& rhs)
		{
			mF8[0] = _mm256_div_ps(mF8[0], rhs.mF8[0]);
			mF8[1] = _mm256_div_ps(mF8[1], rhs.mF8[1]);
			return *this;
		}
		
	public:
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator== (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmpeq_ps(mF8[0], rhs.mF8[0]),
				_mm256_cmpeq_ps(mF8[1], rhs.mF8[1])
			);
		}

		VCL_STRONG_INLINE VectorScalar<bool, 16> operator< (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmplt_ps(mF8[0], rhs.mF8[0]),
				_mm256_cmplt_ps(mF8[1], rhs.mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator<= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmple_ps(mF8[0], rhs.mF8[0]),
				_mm256_cmple_ps(mF8[1], rhs.mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator> (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmpgt_ps(mF8[0], rhs.mF8[0]),
				_mm256_cmpgt_ps(mF8[1], rhs.mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator>= (const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm256_cmpge_ps(mF8[0], rhs.mF8[0]),
				_mm256_cmpge_ps(mF8[1], rhs.mF8[1])
			);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> abs()  const { return VectorScalar<float, 16>(_mm256_abs_ps (mF8[0]), _mm256_abs_ps (mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sin()  const { return VectorScalar<float, 16>(_mm256_sin_ps (mF8[0]), _mm256_sin_ps (mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> cos()  const { return VectorScalar<float, 16>(_mm256_cos_ps (mF8[0]), _mm256_cos_ps (mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> exp()  const { return VectorScalar<float, 16>(_mm256_exp_ps (mF8[0]), _mm256_exp_ps (mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> log()  const { return VectorScalar<float, 16>(_mm256_log_ps (mF8[0]), _mm256_log_ps (mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sgn()  const { return VectorScalar<float, 16>(_mm256_sgn_ps (mF8[0]), _mm256_sgn_ps (mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> sqrt() const { return VectorScalar<float, 16>(_mm256_sqrt_ps(mF8[0]), _mm256_sqrt_ps(mF8[1])); }
		VCL_STRONG_INLINE VectorScalar<float, 16> rcp()  const { return VectorScalar<float, 16>(_mm256_rcp_ps (mF8[0]), _mm256_rcp_ps (mF8[1])); }

		VCL_STRONG_INLINE VectorScalar<float, 16> acos() const { return VectorScalar<float, 16>(_mm256_acos_ps(mF8[0]), _mm256_acos_ps(mF8[1])); }

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16> min(const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				_mm256_min_ps(mF8[0], rhs.mF8[0]),
				_mm256_min_ps(mF8[1], rhs.mF8[1])
			);
		}
		VCL_STRONG_INLINE VectorScalar<float, 16> max(const VectorScalar<float, 16>& rhs) const
		{
			return VectorScalar<float, 16>
			(
				_mm256_max_ps(mF8[0], rhs.mF8[0]),
				_mm256_max_ps(mF8[1], rhs.mF8[1])
			);
		}

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 16>& rhs);
		friend VectorScalar<float, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<float, 16>& a, const VectorScalar<float, 16>& b);

	private:
		__m256 mF8[2];
	};

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 16>& rhs)
	{
		float VCL_ALIGN(16) vars[8];
		_mm256_store_ps(vars + 0, rhs.mF8[0]);
		_mm256_store_ps(vars + 4, rhs.mF8[1]);
		
		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3]
				 << vars[4] << "," << vars[5] << "," << vars[6] << "," << vars[7] << "'";

		return s;
	}

	VCL_STRONG_INLINE VectorScalar<float, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<float, 16>& a, const VectorScalar<float, 16>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<float, 16>
		(
			_mm256_xor_ps(b.mF8[0], _mm256_and_ps(mask.mF8[0], _mm256_xor_ps(b.mF8[0], a.mF8[0]))),
			_mm256_xor_ps(b.mF8[1], _mm256_and_ps(mask.mF8[1], _mm256_xor_ps(b.mF8[1], a.mF8[1])))
		);
	}
}
