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
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl
{
	template<>
	class VectorScalar<bool, 16>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(bool s)
		{
			mF4[0] = s ? _mm_castsi128_ps(_mm_set1_epi32(-1)) : _mm_castsi128_ps(_mm_set1_epi32(0));
			mF4[1] = s ? _mm_castsi128_ps(_mm_set1_epi32(-1)) : _mm_castsi128_ps(_mm_set1_epi32(0));
			mF4[2] = s ? _mm_castsi128_ps(_mm_set1_epi32(-1)) : _mm_castsi128_ps(_mm_set1_epi32(0));
			mF4[3] = s ? _mm_castsi128_ps(_mm_set1_epi32(-1)) : _mm_castsi128_ps(_mm_set1_epi32(0));
		}
		explicit VCL_STRONG_INLINE VectorScalar(const __m128& F4_0, const __m128& F4_1, const __m128& F4_2, const __m128& F4_3)
		{
			mF4[0] = F4_0;
			mF4[1] = F4_1;
			mF4[2] = F4_2;
			mF4[3] = F4_3;
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m128i I4_0, __m128i I4_1, __m128i I4_2, const __m128i& I4_3)
		{
			mF4[0] = _mm_castsi128_ps(I4_0);
			mF4[1] = _mm_castsi128_ps(I4_1);
			mF4[2] = _mm_castsi128_ps(I4_2);
			mF4[3] = _mm_castsi128_ps(I4_3);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator&& (const VectorScalar<bool, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_and_ps(mF4[0], rhs.mF4[0]),
				_mm_and_ps(mF4[1], rhs.mF4[1]),
				_mm_and_ps(mF4[2], rhs.mF4[2]),
				_mm_and_ps(mF4[3], rhs.mF4[3])
			);
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16> operator|| (const VectorScalar<bool, 16>& rhs) const
		{
			return VectorScalar<bool, 16>
			(
				_mm_or_ps(mF4[0], rhs.mF4[0]),
				_mm_or_ps(mF4[1], rhs.mF4[1]),
				_mm_or_ps(mF4[2], rhs.mF4[2]),
				_mm_or_ps(mF4[3], rhs.mF4[3])
			);
		}
		
		VCL_STRONG_INLINE VectorScalar<bool, 16>& operator&= (const VectorScalar<bool, 16>& rhs)
		{
			mF4[0] = _mm_and_ps(mF4[0], rhs.mF4[0]);
			mF4[1] = _mm_and_ps(mF4[1], rhs.mF4[1]);
			mF4[2] = _mm_and_ps(mF4[2], rhs.mF4[2]);
			mF4[3] = _mm_and_ps(mF4[3], rhs.mF4[3]);
			return *this;
		}
		VCL_STRONG_INLINE VectorScalar<bool, 16>& operator|= (const VectorScalar<bool, 16>& rhs)
		{
			mF4[0] = _mm_or_ps(mF4[0], rhs.mF4[0]);
			mF4[1] = _mm_or_ps(mF4[1], rhs.mF4[1]);
			mF4[2] = _mm_or_ps(mF4[2], rhs.mF4[2]);
			mF4[3] = _mm_or_ps(mF4[3], rhs.mF4[3]);
			return *this;
		}

	public:
		friend VectorScalar<float, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<float, 16>& a, const VectorScalar<float, 16>& b);
		friend VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b);
		friend bool any(const VectorScalar<bool, 16>& b);
		friend bool all(const VectorScalar<bool, 16>& b);
		friend bool none(const VectorScalar<bool, 16>& b);

	private:
		__m128 mF4[4];
	};

	VCL_STRONG_INLINE bool any(const VectorScalar<bool, 16>& b)
	{
		int mask  = _mm_movemask_ps(b.mF4[3]) << 12;
			mask |= _mm_movemask_ps(b.mF4[2]) <<  8;
			mask |= _mm_movemask_ps(b.mF4[1]) <<  4;
			mask |= _mm_movemask_ps(b.mF4[0]);

		return mask != 0;
	}

	VCL_STRONG_INLINE bool all(const VectorScalar<bool, 16>& b)
	{
		int mask  = _mm_movemask_ps(b.mF4[3]) << 12;
			mask |= _mm_movemask_ps(b.mF4[2]) <<  8;
			mask |= _mm_movemask_ps(b.mF4[1]) <<  4;
			mask |= _mm_movemask_ps(b.mF4[0]);

		return static_cast<unsigned int>(mask) == 0xffff;
	}

	VCL_STRONG_INLINE bool none(const VectorScalar<bool, 16>& b)
	{
		int mask  = _mm_movemask_ps(b.mF4[3]) << 12;
			mask |= _mm_movemask_ps(b.mF4[2]) <<  8;
			mask |= _mm_movemask_ps(b.mF4[1]) <<  4;
			mask |= _mm_movemask_ps(b.mF4[0]);

		return static_cast<unsigned int>(mask) == 0x0;
	}
}
