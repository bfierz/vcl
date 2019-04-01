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
#include <vcl/core/simd/bool16_sse.h>
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 16> : protected Core::Simd::VectorScalarBase<float, 16, Core::Simd::SimdExt::SSE>
	{
	public:
		using Base = Core::Simd::VectorScalarBase<float, 16, Core::Simd::SimdExt::SSE>;
		using Scalar = float;
		using Self = VectorScalar<float, 16>;
		using Bool = VectorScalar<bool, 16>;
		
		using Base::operator[];
		using Base::get;

		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(float s)
		{
			set(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar
		(
			float s00, float s01, float s02, float s03, float s04, float s05, float s06, float s07,
			float s08, float s09, float s10, float s11, float s12, float s13, float s14, float s15
		)
		{
			set(s00, s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s14, s15);
		}
		explicit VCL_STRONG_INLINE VectorScalar(const __m128& F4_0, const __m128& F4_1, const __m128& F4_2, const __m128& F4_3)
		{
			set(F4_0, F4_1, F4_2, F4_3);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 16>& operator= (const VectorScalar<float, 16>& rhs)
		{
			set(rhs.get(0), rhs.get(1), rhs.get(2), rhs.get(3));
			return *this;
		}

	public:
		VCL_SIMD_BINARY_OP(operator+, _mm_add_ps, 4);
		VCL_SIMD_BINARY_OP(operator-, _mm_sub_ps, 4);
		VCL_SIMD_BINARY_OP(operator*, _mm_mul_ps, 4);
		VCL_SIMD_BINARY_OP(operator/, _mm_div_ps, 4);

	public:
		VCL_SIMD_ASSIGN_OP(operator+=, _mm_add_ps, 4);
		VCL_SIMD_ASSIGN_OP(operator-=, _mm_sub_ps, 4);
		VCL_SIMD_ASSIGN_OP(operator*=, _mm_mul_ps, 4);
		VCL_SIMD_ASSIGN_OP(operator/=, _mm_div_ps, 4);

	public:
		VCL_SIMD_COMP_OP(operator==, _mm_cmpeq_ps, 4);
		VCL_SIMD_COMP_OP(operator!=, _mm_cmpneq_ps, 4);
		VCL_SIMD_COMP_OP(operator<, _mm_cmplt_ps, 4);
		VCL_SIMD_COMP_OP(operator<=, _mm_cmple_ps, 4);
		VCL_SIMD_COMP_OP(operator>, _mm_cmpgt_ps, 4);
		VCL_SIMD_COMP_OP(operator>=, _mm_cmpge_ps, 4);

	public:
		VCL_SIMD_UNARY_OP(abs, Core::Simd::SSE::abs, 4);
		VCL_SIMD_UNARY_OP(sgn, _mm_sgn_ps, 4);

		VCL_SIMD_UNARY_OP(sin, _mm_sin_ps, 4);
		VCL_SIMD_UNARY_OP(cos, _mm_cos_ps, 4);
		VCL_SIMD_UNARY_OP(acos, _mm_acos_ps, 4);

		VCL_SIMD_UNARY_OP(exp, _mm_exp_ps, 4);
		VCL_SIMD_UNARY_OP(log, _mm_log_ps, 4);
		VCL_SIMD_UNARY_OP(sqrt, _mm_sqrt_ps, 4);
		VCL_SIMD_UNARY_OP(rcp, _mmVCL_rcp_ps, 4);
		VCL_SIMD_UNARY_OP(rsqrt, _mmVCL_rsqrt_ps, 4);
		
		VCL_SIMD_QUERY_OP(isinf, _mm_isinf_ps, 4);

	public:
		VCL_SIMD_BINARY_OP(min, _mm_min_ps, 4);
		VCL_SIMD_BINARY_OP(max, _mm_max_ps, 4);
		
		VCL_SIMD_BINARY_REDUCTION_OP(dot, _mmVCL_dp_ps, Core::Simd::Details::add, 4)
		
		VCL_SIMD_UNARY_REDUCTION_OP(min, _mmVCL_hmin_ps, Mathematics::min, 4)
		VCL_SIMD_UNARY_REDUCTION_OP(max, _mmVCL_hmax_ps, Mathematics::max, 4)
	};

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 16>& rhs)
	{
		alignas(16) float vars[16];
		_mm_store_ps(vars +  0, rhs.get(0));
		_mm_store_ps(vars +  4, rhs.get(1));
		_mm_store_ps(vars +  8, rhs.get(2));
		_mm_store_ps(vars + 12, rhs.get(3));

		s << "'" << vars[ 0] << ", " << vars[ 1] << ", " << vars[ 2] << ", " << vars[ 3]
		         << vars[ 4] << ", " << vars[ 5] << ", " << vars[ 6] << ", " << vars[ 7]
				 << vars[ 8] << ", " << vars[ 9] << ", " << vars[10] << ", " << vars[11]
				 << vars[12] << ", " << vars[13] << ", " << vars[14] << ", " << vars[15] << "'";

		return s;
	}

	VCL_STRONG_INLINE VectorScalar<float, 16> select(const VectorScalar<bool, 16>& mask, const VectorScalar<float, 16>& a, const VectorScalar<float, 16>& b)
	{
		return VectorScalar<float, 16>
		(
			Core::Simd::SSE::blend(b.get(0), a.get(0), mask.get(0)),
			Core::Simd::SSE::blend(b.get(1), a.get(1), mask.get(1)),
			Core::Simd::SSE::blend(b.get(2), a.get(2), mask.get(2)),
			Core::Simd::SSE::blend(b.get(3), a.get(3), mask.get(3))
		);
	}
}
