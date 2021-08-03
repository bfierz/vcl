/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#include <vcl/core/simd/bool4_neon.h>
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/intrinsics_neon.h>
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl {
	template<>
	class alignas(16) VectorScalar<float, 4> : protected Core::Simd::VectorScalarBase<float, 4, Core::Simd::SimdExt::NEON>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(NEON)

	public:
		VCL_SIMD_BINARY_OP(operator+, vaddq_f32, 1)
		VCL_SIMD_BINARY_OP(operator-, vsubq_f32, 1)
		VCL_SIMD_BINARY_OP(operator*, vmulq_f32, 1)
		VCL_SIMD_BINARY_OP(operator/, vdivq_f32, 1)

	public:
		VCL_SIMD_ASSIGN_OP(operator+=, vaddq_f32, 1)
		VCL_SIMD_ASSIGN_OP(operator-=, vsubq_f32, 1)
		VCL_SIMD_ASSIGN_OP(operator*=, vmulq_f32, 1)
		VCL_SIMD_ASSIGN_OP(operator/=, vdivq_f32, 1)

	public:
		VCL_SIMD_COMP_OP(operator==, vceqq_f32, 1)
		VCL_SIMD_COMP_OP(operator!=, vcneqq_f32, 1)
		VCL_SIMD_COMP_OP(operator<, vcltq_f32, 1)
		VCL_SIMD_COMP_OP(operator<=, vcleq_f32, 1)
		VCL_SIMD_COMP_OP(operator>, vcgtq_f32, 1)
		VCL_SIMD_COMP_OP(operator>=, vcgeq_f32, 1)

	public:
		VCL_SIMD_UNARY_OP(abs, vabsq_f32, 1)
		VCL_SIMD_UNARY_OP(sin, vsinq_f32, 1)
		VCL_SIMD_UNARY_OP(cos, vcosq_f32, 1)
		VCL_SIMD_UNARY_OP(exp, vexpq_f32, 1)
		VCL_SIMD_UNARY_OP(log, vlogq_f32, 1)
		VCL_SIMD_UNARY_OP(sgn, vsgnq_f32, 1)
		VCL_SIMD_UNARY_OP(sqrt, vsqrtq_f32, 1)
		VCL_SIMD_UNARY_OP(rcp, vrcpq_f32, 1)
		VCL_SIMD_UNARY_OP(rsqrt, vrsqrtq_f32, 1)

		VCL_SIMD_UNARY_OP(acos, vacosq_f32, 1)

		VCL_SIMD_QUERY_OP(isinf, visinfq_f32, 1)

	public:
		VCL_SIMD_BINARY_OP(min, vminq_f32, 1)
		VCL_SIMD_BINARY_OP(max, vmaxq_f32, 1)

		VCL_SIMD_BINARY_REDUCTION_OP(dot, vdotq_f32, VCL_UNUSED, 1)

		VCL_SIMD_UNARY_REDUCTION_OP(min, vpminq_f32, VCL_UNUSED, 1)
		VCL_SIMD_UNARY_REDUCTION_OP(max, vpmaxq_f32, VCL_UNUSED, 1)
	};

	VCL_STRONG_INLINE VectorScalar<float, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<float, 4>& a, const VectorScalar<float, 4>& b) noexcept
	{
		return VectorScalar<float, 4>(vbslq_f32(mask.get(0), a.get(0), b.get(0)));
	}

	VCL_STRONG_INLINE std::ostream& operator<<(std::ostream& s, const VectorScalar<float, 4>& rhs)
	{
		alignas(8) float vars[4];
		vst1q_f32(vars + 0, rhs.get(0));

		s << "'" << vars[0] << "," << vars[1] << "," << vars[2] << "," << vars[3] << "'";
		return s;
	}
}
