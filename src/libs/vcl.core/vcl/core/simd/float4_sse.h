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
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/intrinsics_sse.h>
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl {
	template<>
	class VectorScalar<float, 4> : protected Core::Simd::VectorScalarBase<float, 4, Core::Simd::SimdExt::SSE>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(SSE)

	public:
		VCL_SIMD_BINARY_OP(operator+, _mm_add_ps, 1)
		VCL_SIMD_BINARY_OP(operator-, _mm_sub_ps, 1)
		VCL_SIMD_BINARY_OP(operator*, _mm_mul_ps, 1)
		VCL_SIMD_BINARY_OP(operator/, _mm_div_ps, 1)
		
	public:
		VCL_SIMD_ASSIGN_OP(operator+=, _mm_add_ps, 1)
		VCL_SIMD_ASSIGN_OP(operator-=, _mm_sub_ps, 1)
		VCL_SIMD_ASSIGN_OP(operator*=, _mm_mul_ps, 1)
		VCL_SIMD_ASSIGN_OP(operator/=, _mm_div_ps, 1)
		
	public:
		VCL_SIMD_COMP_OP(operator==, _mm_cmpeq_ps,  1)
		VCL_SIMD_COMP_OP(operator!=, _mm_cmpneq_ps, 1)
		VCL_SIMD_COMP_OP(operator<,  _mm_cmplt_ps,  1)
		VCL_SIMD_COMP_OP(operator<=, _mm_cmple_ps,  1)
		VCL_SIMD_COMP_OP(operator>,  _mm_cmpgt_ps,  1)
		VCL_SIMD_COMP_OP(operator>=, _mm_cmpge_ps,  1)

	public:
		VCL_SIMD_UNARY_OP(abs, Core::Simd::SSE::abs_f32, 1)
		VCL_SIMD_UNARY_OP(sgn, Core::Simd::SSE::sgn_f32, 1)

		VCL_SIMD_UNARY_OP(sin, _mm_sin_ps, 1)
		VCL_SIMD_UNARY_OP(cos, _mm_cos_ps, 1)
		VCL_SIMD_UNARY_OP(acos, _mm_acos_ps, 1)

		VCL_SIMD_UNARY_OP(exp, _mm_exp_ps, 1)
		VCL_SIMD_UNARY_OP(log, _mm_log_ps, 1)
		VCL_SIMD_UNARY_OP(sqrt, _mm_sqrt_ps, 1)
		VCL_SIMD_UNARY_OP(rcp, _mmVCL_rcp_ps, 1)
		VCL_SIMD_UNARY_OP(rsqrt, _mmVCL_rsqrt_ps, 1)
		
		VCL_SIMD_QUERY_OP(isinf, _mm_isinf_ps, 1)

	public:
		VCL_SIMD_BINARY_OP(min, _mm_min_ps, 1)
		VCL_SIMD_BINARY_OP(max, _mm_max_ps, 1)

		VCL_SIMD_BINARY_REDUCTION_OP(dot, _mmVCL_dp_ps, VCL_UNUSED, 1)

		VCL_SIMD_UNARY_REDUCTION_OP(min, _mmVCL_hmin_ps, VCL_UNUSED, 1)
		VCL_SIMD_UNARY_REDUCTION_OP(max, _mmVCL_hmax_ps, VCL_UNUSED, 1)
	};

	VCL_STRONG_INLINE VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b) noexcept
	{
		return VectorScalar<float, 4>(Core::Simd::SSE::blend_f32(b.get(0), a.get(0), mask.get(0)));
	}

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 4>& rhs)
	{
		alignas(16) float vars[4];
		_mm_store_ps(vars + 0, rhs.get(0));
		
		s << "'" << vars[0] << ", " << vars[1] << ", " << vars[2] << ", " << vars[3] << "'";
		return s;
	}
}
