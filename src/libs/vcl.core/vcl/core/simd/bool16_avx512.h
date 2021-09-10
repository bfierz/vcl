/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2021 Basil Fierz
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
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl {
	template<>
	class VectorScalar<bool, 16> : protected Core::Simd::VectorScalarBase<bool, 16, Core::Simd::SimdExt::AVX512>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(AVX512)

	public:
		VCL_SIMD_BINARY_OP(operator&&, _mm512_kand, 1)
		VCL_SIMD_BINARY_OP(operator||, _mm512_kor, 1)

		VCL_SIMD_ASSIGN_OP(operator&=, _mm512_kand, 1)
		VCL_SIMD_ASSIGN_OP(operator|=, _mm512_kor, 1)
	};

	VCL_STRONG_INLINE bool any(const VectorScalar<bool, 16>& b)
	{
		const auto mask = _cvtmask16_u32(b.get(0));
		return mask != 0;
	}

	VCL_STRONG_INLINE bool all(const VectorScalar<bool, 16>& b)
	{
		const auto mask = _cvtmask16_u32(b.get(0));
		return mask == 0xffff;
	}

	VCL_STRONG_INLINE bool none(const VectorScalar<bool, 16>& b)
	{
		const auto mask = _cvtmask16_u32(b.get(0));
		return mask == 0x0;
	}
}
