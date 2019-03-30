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
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_sse.h>

namespace Vcl
{
	template<>
	class VectorScalar<float, 8> : protected Core::Simd::VectorScalarBase<float, 8, Core::Simd::SimdExt::SSE>
	{
	public:
		using Self = VectorScalar<float, 8>;
		using Bool = VectorScalar<bool, 8>;
		
		using Core::Simd::VectorScalarBase<float, 8, Core::Simd::SimdExt::SSE>::get;

		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(float s)
		{
			set(s);
		}
		explicit VCL_STRONG_INLINE VectorScalar(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7)
		{
			set(s0, s1, s2, s3, s4, s5, s6, s7);
		}
		VCL_STRONG_INLINE explicit VectorScalar(__m128 F4_0, __m128 F4_1)
		{
			set(F4_0, F4_1);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 8>& operator = (const VectorScalar<float, 8>& rhs)
		{
			set(rhs.get(0), rhs.get(1));
			return *this;
		}

	public:
		VCL_STRONG_INLINE float operator[] (int idx) const
		{
			VclRequire(0 <= idx && idx < 8, "Access is in range.");
			return _mmVCL_extract_ps(get(idx / 4), idx % 4);
		}

	public:
		VCL_STRONG_INLINE VectorScalar<float, 8> operator- () const
		{
			return (*this) * VectorScalar<float, 8>(-1);
		}

	public:
		VCL_SIMD_BINARY_OP(operator+, _mm_add_ps, 2);
		VCL_SIMD_BINARY_OP(operator-, _mm_sub_ps, 2);
		VCL_SIMD_BINARY_OP(operator*, _mm_mul_ps, 2);
		VCL_SIMD_BINARY_OP(operator/, _mm_div_ps, 2);
		
	public:
		VCL_SIMD_ASSIGN_OP(operator+=, _mm_add_ps, 2);
		VCL_SIMD_ASSIGN_OP(operator-=, _mm_sub_ps, 2);
		VCL_SIMD_ASSIGN_OP(operator*=, _mm_mul_ps, 2);
		VCL_SIMD_ASSIGN_OP(operator/=, _mm_div_ps, 2);
		
	public:
		VCL_SIMD_COMP_OP(operator==, _mm_cmpeq_ps,  2);
		VCL_SIMD_COMP_OP(operator!=, _mm_cmpneq_ps, 2);
		VCL_SIMD_COMP_OP(operator<,  _mm_cmplt_ps,  2);
		VCL_SIMD_COMP_OP(operator<=, _mm_cmple_ps,  2);
		VCL_SIMD_COMP_OP(operator>,  _mm_cmpgt_ps,  2);
		VCL_SIMD_COMP_OP(operator>=, _mm_cmpge_ps,  2);

	public:
		VCL_SIMD_UNARY_OP(abs, _mm_abs_ps, 2);
		VCL_SIMD_UNARY_OP(sgn, _mm_sgn_ps, 2);

		VCL_SIMD_UNARY_OP(sin, _mm_sin_ps, 2);
		VCL_SIMD_UNARY_OP(cos, _mm_cos_ps, 2);
		VCL_SIMD_UNARY_OP(acos, _mm_acos_ps, 2);

		VCL_SIMD_UNARY_OP(exp, _mm_exp_ps, 2);
		VCL_SIMD_UNARY_OP(log, _mm_log_ps, 2);
		VCL_SIMD_UNARY_OP(sqrt, _mm_sqrt_ps, 2);
		VCL_SIMD_UNARY_OP(rcp, _mmVCL_rcp_ps, 2);
		VCL_SIMD_UNARY_OP(rsqrt, _mmVCL_rsqrt_ps, 2);

	public:
		VCL_SIMD_BINARY_OP(min, _mm_min_ps, 2);
		VCL_SIMD_BINARY_OP(max, _mm_max_ps, 2);

		VCL_STRONG_INLINE float dot(const VectorScalar<float, 8>& rhs) const
		{
			return 
				_mmVCL_dp_ps(get(0), rhs.get(0)) + 
				_mmVCL_dp_ps(get(1), rhs.get(1));
		}

		VCL_STRONG_INLINE float min() const
		{
			return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(_mmVCL_hmin_ps(get(0))), _mm_set_ss(_mmVCL_hmin_ps(get(1)))));
		}
		VCL_STRONG_INLINE float max() const
		{
			return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(_mmVCL_hmax_ps(get(0))), _mm_set_ss(_mmVCL_hmax_ps(get(1)))));
		}
	};

	VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<float, 8>& rhs)
	{
		alignas(16) float vars[8];
		_mm_store_ps(vars + 0, rhs.get(0));
		_mm_store_ps(vars + 4, rhs.get(1));
		
		s << "'" << vars[0] << ", " << vars[1] << ", " << vars[2] << ", " << vars[3]
				 << vars[4] << ", " << vars[5] << ", " << vars[6] << ", " << vars[7] << "'";

		return s;
	}

	VCL_STRONG_INLINE VectorScalar<float, 8> select(const VectorScalar<bool, 8>& mask, const VectorScalar<float, 8>& a, const VectorScalar<float, 8>& b)
	{
#ifdef VCL_VECTORIZE_SSE4_1
		// SSE way
		return VectorScalar<float, 8>
		(
			_mm_blendv_ps(b.get(0), a.get(0), mask.mF4[0]),
			_mm_blendv_ps(b.get(1), a.get(1), mask.mF4[1])
		);
#else
		// (((b ^ a) & mask)^b)
		return VectorScalar<float, 8>
		(
			_mm_xor_ps(b.get(0), _mm_and_ps(mask.mF4[0], _mm_xor_ps(b.get(0), a.get(0)))),
			_mm_xor_ps(b.get(1), _mm_and_ps(mask.mF4[1], _mm_xor_ps(b.get(1), a.get(1))))
		);
#endif
	}

	VCL_STRONG_INLINE VectorScalar<bool, 8> isinf(const VectorScalar<float, 8>& x)
	{
		return VectorScalar<bool, 8>
		(
			_mm_isinf_ps(x.get(0)),
			_mm_isinf_ps(x.get(1))
		);
	}
}
