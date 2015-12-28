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
	class VectorScalar<bool, 8>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(bool s)
		{
			mF8 = s ? _mm256_castsi256_ps(_mm256_set1_epi32(-1)) : _mm256_castsi256_ps(_mm256_set1_epi32(0));
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m256 F8) : mF8(F8) { }
		explicit VCL_STRONG_INLINE VectorScalar(__m256i I8) : mF8(_mm256_castsi256_ps(I8)) {}

	public:
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator&& (const VectorScalar<bool, 8>& rhs)
		{
			return VectorScalar<bool, 8>(_mm256_and_ps(mF8, rhs.mF8));
		}
		VCL_STRONG_INLINE VectorScalar<bool, 8> operator|| (const VectorScalar<bool, 8>& rhs)
		{
			return VectorScalar<bool, 8>(_mm256_or_ps(mF8, rhs.mF8));
		}

		VCL_STRONG_INLINE VectorScalar<bool, 8>& operator&= (const VectorScalar<bool, 8>& rhs) { mF8 = _mm256_and_ps(mF8, rhs.mF8); return *this; }
		VCL_STRONG_INLINE VectorScalar<bool, 8>& operator|= (const VectorScalar<bool, 8>& rhs) { mF8 = _mm256_or_ps(mF8, rhs.mF8);  return *this; }

	public:
		friend VectorScalar<float, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<float, 8>& a, const VectorScalar<float, 8>& b);
		friend VectorScalar<int, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<int, 8>& a, const VectorScalar<int, 8>& b);
		friend bool any(const VectorScalar<bool, 8>& b);
		friend bool all(const VectorScalar<bool, 8>& b);
		friend bool none(const VectorScalar<bool, 8>& b);

	private:
		__m256 mF8;
	};

	VCL_STRONG_INLINE bool any(const VectorScalar<bool, 8>& b)
	{
		return _mm256_movemask_ps(b.mF8) != 0;
	}

	VCL_STRONG_INLINE bool all(const VectorScalar<bool, 8>& b)
	{
		return static_cast<unsigned int>(_mm256_movemask_ps(b.mF8)) == 0xff;
	}

	VCL_STRONG_INLINE bool none(const VectorScalar<bool, 8>& b)
	{
		return static_cast<unsigned int>(_mm256_movemask_ps(b.mF8)) == 0x0;
	}
}
