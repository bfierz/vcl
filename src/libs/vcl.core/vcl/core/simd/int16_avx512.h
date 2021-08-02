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
#include <vcl/core/simd/bool16_avx512.h>
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/intrinsics_avx512.h>
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl {
	template<>
	class alignas(64) VectorScalar<int, 16> : protected Core::Simd::VectorScalarBase<int, 16, Core::Simd::SimdExt::AVX512>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(AVX512)

	public:
		VCL_SIMD_BINARY_OP(operator+, _mm512_add_epi32, 1)
		VCL_SIMD_BINARY_OP(operator-, _mm512_sub_epi32, 1)
		VCL_SIMD_BINARY_OP(operator*, _mm512_mullo_epi32, 1)

	public:
		VCL_SIMD_ASSIGN_OP(operator+=, _mm512_add_epi32, 1)
		VCL_SIMD_ASSIGN_OP(operator-=, _mm512_sub_epi32, 1)
		VCL_SIMD_ASSIGN_OP(operator*=, _mm512_mullo_epi32, 1)

	public:
		VCL_SIMD_COMP_OP(operator==, _mm512_cmpeq_epi32_mask,  1)
		VCL_SIMD_COMP_OP(operator!=, _mm512_cmpneq_epi32_mask, 1)
		VCL_SIMD_COMP_OP(operator<,  _mm512_cmplt_epi32_mask,  1)
		VCL_SIMD_COMP_OP(operator<=, _mm512_cmple_epi32_mask,  1)
		VCL_SIMD_COMP_OP(operator>,  _mm512_cmpgt_epi32_mask,  1)
		VCL_SIMD_COMP_OP(operator>=, _mm512_cmpge_epi32_mask,  1)

	public:
		VCL_SIMD_UNARY_OP(abs, _mm512_abs_epi32, 1)

	public:
		VCL_SIMD_BINARY_OP(operator&, _mm512_and_si512, 1)
		VCL_SIMD_BINARY_OP(operator|, _mm512_or_si512, 1)

		VCL_SIMD_BINARY_OP(min, _mm512_min_epi32, 1)
		VCL_SIMD_BINARY_OP(max, _mm512_max_epi32, 1)
	};

	VCL_STRONG_INLINE VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b)
	{
		return VectorScalar<int, 16>
		(
			_mm512_mask_blend_epi32(mask.get(0), b.get(0), a.get(0))
		);
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 16>& rhs)
	{
		alignas(64) int vars[16];
		_mm512_store_si512(reinterpret_cast<__m512i*>(vars + 0), rhs.get(0));

		s << "'" << vars[ 0] << ", " << vars[ 1] << ", " << vars[ 2] << ", " << vars[ 3] << ", "
				 << vars[ 4] << ", " << vars[ 5] << ", " << vars[ 6] << ", " << vars[ 7] << ", "
				 << vars[ 8] << ", " << vars[ 9] << ", " << vars[10] << ", " << vars[11] << ", "
				 << vars[12] << ", " << vars[13] << ", " << vars[14] << ", " << vars[15] << "'";

		return s;
	}
}
