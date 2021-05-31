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
#include <vcl/core/simd/bool8_sse.h>
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/intrinsics_sse.h>
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 8> : protected Core::Simd::VectorScalarBase<float, 8, Core::Simd::SimdExt::SSE>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(SSE)

	public:
		VCL_SIMD_BINARY_OP(operator+, _mm_add_ps, 2)
		VCL_SIMD_BINARY_OP(operator-, _mm_sub_ps, 2)
		VCL_SIMD_BINARY_OP(operator*, _mm_mul_ps, 2)
		VCL_SIMD_BINARY_OP(operator/, _mm_div_ps, 2)
		
	public:
		VCL_SIMD_ASSIGN_OP(operator+=, _mm_add_ps, 2)
		VCL_SIMD_ASSIGN_OP(operator-=, _mm_sub_ps, 2)
		VCL_SIMD_ASSIGN_OP(operator*=, _mm_mul_ps, 2)
		VCL_SIMD_ASSIGN_OP(operator/=, _mm_div_ps, 2)
		
	public:
		VCL_SIMD_COMP_OP(operator==, _mm_cmpeq_ps,  2)
		VCL_SIMD_COMP_OP(operator!=, _mm_cmpneq_ps, 2)
		VCL_SIMD_COMP_OP(operator<,  _mm_cmplt_ps,  2)
		VCL_SIMD_COMP_OP(operator<=, _mm_cmple_ps,  2)
		VCL_SIMD_COMP_OP(operator>,  _mm_cmpgt_ps,  2)
		VCL_SIMD_COMP_OP(operator>=, _mm_cmpge_ps,  2)

	public:
		VCL_SIMD_UNARY_OP(abs, Core::Simd::SSE::abs_f32, 2)
		VCL_SIMD_UNARY_OP(sgn, Core::Simd::SSE::sgn_f32, 2)

		VCL_SIMD_UNARY_OP(sin, _mm_sin_ps, 2)
		VCL_SIMD_UNARY_OP(cos, _mm_cos_ps, 2)
		VCL_SIMD_UNARY_OP(acos, _mm_acos_ps, 2)

		VCL_SIMD_UNARY_OP(exp, _mm_exp_ps, 2)
		VCL_SIMD_UNARY_OP(log, _mm_log_ps, 2)
		VCL_SIMD_UNARY_OP(sqrt, _mm_sqrt_ps, 2)
		VCL_SIMD_UNARY_OP(rcp, _mmVCL_rcp_ps, 2)
		VCL_SIMD_UNARY_OP(rsqrt, _mmVCL_rsqrt_ps, 2)
		
		VCL_SIMD_QUERY_OP(isinf, _mm_isinf_ps, 2)

	public:
		VCL_SIMD_BINARY_OP(min, _mm_min_ps, 2)
		VCL_SIMD_BINARY_OP(max, _mm_max_ps, 2)
		
		VCL_SIMD_BINARY_REDUCTION_OP(dot, _mmVCL_dp_ps, Core::Simd::Details::add, 2)
		
		VCL_SIMD_UNARY_REDUCTION_OP(min, _mmVCL_hmin_ps, Mathematics::min, 2)
		VCL_SIMD_UNARY_REDUCTION_OP(max, _mmVCL_hmax_ps, Mathematics::max, 2)
	};

	VCL_STRONG_INLINE VectorScalar<float, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<float, 8>& a, const VectorScalar<float, 8>& b) noexcept
	{
		return VectorScalar<float, 8>
		(
			Core::Simd::SSE::blend_f32(b.get(0), a.get(0), mask.get(0)),
			Core::Simd::SSE::blend_f32(b.get(1), a.get(1), mask.get(1))
		);
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 8>& rhs)
	{
		alignas(16) float vars[8];
		_mm_store_ps(vars + 0, rhs.get(0));
		_mm_store_ps(vars + 4, rhs.get(1));
		
		s << "'" << vars[0] << ", " << vars[1] << ", " << vars[2] << ", " << vars[3]
				 << vars[4] << ", " << vars[5] << ", " << vars[6] << ", " << vars[7] << "'";

		return s;
	}
}
