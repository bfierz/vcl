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

// VCL 
#include <vcl/core/simd/bool4_sse.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl
{
	template<>
	class VectorScalar<int, 4>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() {}
		VCL_STRONG_INLINE VectorScalar(int s)
		{
			mF4 = _mm_set1_epi32(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar(int s0, int s1, int s2, int s3)
		{
			mF4 = _mm_set_epi32(s3, s2, s1, s0);
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m128i F4) : mF4(F4) {}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 4>& operator= (const VectorScalar<int, 4>& rhs) { mF4 = rhs.mF4; return *this; }

	public:
		int& operator[] (int idx)
		{
			Require(0 <= idx && idx < 4, "Access is in range.");

			return mF4.m128i_i32[idx];
		}

		int operator[] (int idx) const
		{
			Require(0 <= idx && idx < 4, "Access is in range.");

			return mF4.m128i_i32[idx];
		}

		explicit operator __m128i() const
		{
			return mF4;
		}

	public:
		VCL_STRONG_INLINE VectorScalar<int, 4> operator+ (const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mm_add_epi32(mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<int, 4> operator- (const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mm_sub_epi32(mF4, rhs.mF4)); }
		VCL_STRONG_INLINE VectorScalar<int, 4> operator* (const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mm_mullo_epi32(mF4, rhs.mF4)); }

	public:
		VCL_STRONG_INLINE VectorScalar<int, 4> abs() const { return VectorScalar<int, 4>(_mm_abs_epi32(mF4)); }
		VCL_STRONG_INLINE VectorScalar<int, 4> max(const VectorScalar<int, 4>& rhs) const { return VectorScalar<int, 4>(_mm_max_epi32(mF4, rhs.mF4)); }

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator== (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmpeq_epi32(mF4, rhs.mF4));
		}

		VCL_STRONG_INLINE VectorScalar<bool, 4> operator< (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmplt_epi32(mF4, rhs.mF4));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator<= (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmple_epi32(mF4, rhs.mF4));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator> (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmpgt_epi32(mF4, rhs.mF4));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator>= (const VectorScalar<int, 4>& rhs) const
		{
			return VectorScalar<bool, 4>(_mm_cmpge_epi32(mF4, rhs.mF4));
		}

	public:
		friend std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 4>& rhs);
		friend VectorScalar<int, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<int, 4>& a, const VectorScalar<int, 4>& b);
		friend VectorScalar<int, 4> signum(const VectorScalar<int, 4>& a);

	private:
		__m128i mF4;
	};

	VCL_STRONG_INLINE VectorScalar<int, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<int, 4>& a, const VectorScalar<int, 4>& b)
	{
		// (((b ^ a) & mask)^b)
		return VectorScalar<int, 4>(_mm_xor_si128(b.mF4, _mm_and_si128(_mm_castps_si128(mask.mF4), _mm_xor_si128(b.mF4, a.mF4))));
	}

	VCL_STRONG_INLINE VectorScalar<int, 4> signum(const VectorScalar<int, 4>& a)
	{
		return VectorScalar<int, 4>
		(
			_mm_and_si128
			(
				_mm_or_si128
				(
					_mm_and_si128(a.mF4, _mm_set1_epi32(0x80000000)), _mm_set1_epi32(1)
				), _mm_cmpneq_epi32(a.mF4, _mm_setzero_si128())
			)
		);
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 4>& rhs)
	{
		s << "'" << rhs.mF4.m128i_i32[0] << "," << rhs.mF4.m128i_i32[1] << "," << rhs.mF4.m128i_i32[2] << "," << rhs.mF4.m128i_i32[3] << "'";

		return s;
	}
}
