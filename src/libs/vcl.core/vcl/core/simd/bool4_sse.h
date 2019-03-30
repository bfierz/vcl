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
	class VectorScalar<bool, 4>
	{
	public:
		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(bool s)
		{
			_data[0] = s ? _mm_castsi128_ps(_mm_set1_epi32(-1)) : _mm_castsi128_ps(_mm_set1_epi32(0));
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m128 F4) { _data[0] = F4; }
		explicit VCL_STRONG_INLINE VectorScalar(__m128i I4) { _data[0] = _mm_castsi128_ps(I4); }
		
		VCL_STRONG_INLINE __m128 get(int i = 0) const
		{
			VclRequire(0 == i, "Access is in range.");
			return _data[i];
		}
	public:
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator&& (const VectorScalar<bool, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_and_ps(get(0), rhs.get(0))); }
		VCL_STRONG_INLINE VectorScalar<bool, 4> operator|| (const VectorScalar<bool, 4>& rhs) const { return VectorScalar<bool, 4>(_mm_or_ps (get(0), rhs.get(0))); }

		VCL_STRONG_INLINE VectorScalar<bool, 4>& operator&= (const VectorScalar<bool, 4>& rhs) { _data[0] = _mm_and_ps(get(0), rhs.get(0)); return *this; }
		VCL_STRONG_INLINE VectorScalar<bool, 4>& operator|= (const VectorScalar<bool, 4>& rhs) { _data[0] = _mm_or_ps(get(0), rhs.get(0));  return *this; }

	private:
		__m128 _data[1];
	};

	VCL_STRONG_INLINE bool any(const VectorScalar<bool, 4>& b)
	{
		//int alignas(16) vars[4];
		//_mm_store_ps((float*) vars, b.get(0));
		//
		//return vars[0] | vars[1] | vars[2] | vars[3];

		return _mm_movemask_ps(b.get(0)) != 0;
	}

	VCL_STRONG_INLINE bool all(const VectorScalar<bool, 4>& b)
	{
		//int alignas(16) vars[4];
		//_mm_store_ps((float*) vars, b.get(0));
		//
		//return vars[0] & vars[1] & vars[2] & vars[3];

		return static_cast<unsigned int>(_mm_movemask_ps(b.get(0))) == 0xf;
	}

	VCL_STRONG_INLINE bool none(const VectorScalar<bool, 4>& b)
	{
		//int alignas(16) vars[4];
		//_mm_store_ps((float*) vars, b.get(0));
		//
		//return !(vars[0] | vars[1] | vars[2] | vars[3]);

		return static_cast<unsigned int>(_mm_movemask_ps(b.get(0))) == 0x0;
	}
}
