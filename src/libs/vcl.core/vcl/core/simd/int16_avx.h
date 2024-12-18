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
#include <vcl/core/simd/bool16_avx.h>
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/intrinsics_avx.h>
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl {
	template<>
	class alignas(32) VectorScalar<int, 16> : protected Core::Simd::VectorScalarBase<int, 16, Core::Simd::SimdExt::AVX>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(AVX)

	public:
		VCL_SIMD_BINARY_OP(operator+, _mmVCL_add_epi32, 2)
		VCL_SIMD_BINARY_OP(operator-, _mmVCL_sub_epi32, 2)
		VCL_SIMD_BINARY_OP(operator*, _mmVCL_mullo_epi32, 2)

	public:
		VCL_SIMD_ASSIGN_OP(operator+=, _mmVCL_add_epi32, 2)
		VCL_SIMD_ASSIGN_OP(operator-=, _mmVCL_sub_epi32, 2)
		VCL_SIMD_ASSIGN_OP(operator*=, _mmVCL_mullo_epi32, 2)

	public:
		VCL_SIMD_COMP_OP(operator==, _mmVCL_cmpeq_epi32, 2)
		VCL_SIMD_COMP_OP(operator!=, _mmVCL_cmpneq_epi32, 2)
		VCL_SIMD_COMP_OP(operator<, _mmVCL_cmplt_epi32, 2)
		VCL_SIMD_COMP_OP(operator<=, _mmVCL_cmple_epi32, 2)
		VCL_SIMD_COMP_OP(operator>, _mmVCL_cmpgt_epi32, 2)
		VCL_SIMD_COMP_OP(operator>=, _mmVCL_cmpge_epi32, 2)

	public:
		VCL_SIMD_UNARY_OP(abs, _mmVCL_abs_epi32, 2)

	public:
		VCL_SIMD_BINARY_OP(operator&, _mmVCL_and_si256, 2)
		VCL_SIMD_BINARY_OP(operator|, _mmVCL_or_si256, 2)

		VCL_SIMD_BINARY_OP(min, _mmVCL_min_epi32, 2)
		VCL_SIMD_BINARY_OP(max, _mmVCL_max_epi32, 2)
	};

	VCL_STRONG_INLINE VectorScalar<int, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<int, 16>& a, const VectorScalar<int, 16>& b) noexcept
	{
		return VectorScalar<int, 16>(
			_mmVCL_xor_si256(b.get(0), _mmVCL_and_si256(_mm256_castps_si256(mask.get(0)), _mmVCL_xor_si256(b.get(0), a.get(0)))),
			_mmVCL_xor_si256(b.get(1), _mmVCL_and_si256(_mm256_castps_si256(mask.get(1)), _mmVCL_xor_si256(b.get(1), a.get(1)))));
	}

	VCL_STRONG_INLINE std::ostream& operator<<(std::ostream& s, const VectorScalar<int, 16>& rhs)
	{
		alignas(32) int vars[16];
		_mm256_store_si256(reinterpret_cast<__m256i*>(vars + 0), rhs.get(0));
		_mm256_store_si256(reinterpret_cast<__m256i*>(vars + 8), rhs.get(1));

		s << "'" << vars[0] << ", " << vars[1] << ", " << vars[2] << ", " << vars[3] << ", "
		  << vars[4] << ", " << vars[5] << ", " << vars[6] << ", " << vars[7] << ", "
		  << vars[8] << ", " << vars[9] << ", " << vars[10] << ", " << vars[11] << ", "
		  << vars[12] << ", " << vars[13] << ", " << vars[14] << ", " << vars[15] << "'";

		return s;
	}
}
