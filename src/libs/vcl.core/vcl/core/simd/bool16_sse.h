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
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl
{
	template<>
	class VectorScalar<bool, 16> : protected Core::Simd::VectorScalarBase<bool, 16, Core::Simd::SimdExt::SSE>
	{
	public:
		using Base = Core::Simd::VectorScalarBase<bool, 16, Core::Simd::SimdExt::SSE>;
		using Scalar = bool;
		using Self = VectorScalar<bool, 16>;
		
		using Base::operator[];
		using Base::get;

		VCL_STRONG_INLINE VectorScalar() = default;
		explicit VCL_STRONG_INLINE VectorScalar(bool s) { set(s); }
		explicit VCL_STRONG_INLINE VectorScalar
		(
			bool s00, bool s01, bool s02, bool s03, bool s04, bool s05, bool s06, bool s07,
			bool s08, bool s09, bool s10, bool s11, bool s12, bool s13, bool s14, bool s15
		)
		{
			set(s00, s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s14, s15);
		}
		explicit VCL_STRONG_INLINE VectorScalar(const __m128& F4_0, const __m128& F4_1, const __m128& F4_2, const __m128& F4_3)
		{
			_data[0] = F4_0;
			_data[1] = F4_1;
			_data[2] = F4_2;
			_data[3] = F4_3;
		}
		explicit VCL_STRONG_INLINE VectorScalar(__m128i I4_0, __m128i I4_1, __m128i I4_2, const __m128i& I4_3)
		{
			_data[0] = _mm_castsi128_ps(I4_0);
			_data[1] = _mm_castsi128_ps(I4_1);
			_data[2] = _mm_castsi128_ps(I4_2);
			_data[3] = _mm_castsi128_ps(I4_3);
		}

	public:
		VCL_SIMD_BINARY_OP(operator&&, _mm_and_ps, 4)
		VCL_SIMD_BINARY_OP(operator||, _mm_or_ps, 4)

		VCL_SIMD_ASSIGN_OP(operator&=, _mm_and_ps, 4)
		VCL_SIMD_ASSIGN_OP(operator|=, _mm_or_ps, 4)
	};

	VCL_STRONG_INLINE bool any(const VectorScalar<bool, 16>& b)
	{
		int mask  = _mm_movemask_ps(b.get(3)) << 12;
			mask |= _mm_movemask_ps(b.get(2)) <<  8;
			mask |= _mm_movemask_ps(b.get(1)) <<  4;
			mask |= _mm_movemask_ps(b.get(0));

		return mask != 0;
	}

	VCL_STRONG_INLINE bool all(const VectorScalar<bool, 16>& b)
	{
		int mask  = _mm_movemask_ps(b.get(3)) << 12;
			mask |= _mm_movemask_ps(b.get(2)) <<  8;
			mask |= _mm_movemask_ps(b.get(1)) <<  4;
			mask |= _mm_movemask_ps(b.get(0));

		return static_cast<unsigned int>(mask) == 0xffff;
	}

	VCL_STRONG_INLINE bool none(const VectorScalar<bool, 16>& b)
	{
		int mask  = _mm_movemask_ps(b.get(3)) << 12;
			mask |= _mm_movemask_ps(b.get(2)) <<  8;
			mask |= _mm_movemask_ps(b.get(1)) <<  4;
			mask |= _mm_movemask_ps(b.get(0));

		return static_cast<unsigned int>(mask) == 0x0;
	}
}
