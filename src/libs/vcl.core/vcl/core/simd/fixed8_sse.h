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
#include <vcl/core/simd/common.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/math/fixed.h>

namespace Vcl
{
	template<>
	class VectorScalar<Mathematics::fixed<short, 11>, 8> : protected Core::Simd::VectorScalarBase<short, 8, Core::Simd::SimdExt::SSE>
	{
	public:
		using Base = Core::Simd::VectorScalarBase<short, 8, Core::Simd::SimdExt::SSE>;
		using Self = VectorScalar<Scalar, Base::NrValues>;
		using Bool = VectorScalar<bool, Base::NrValues>;
		using Base::operator[];

		using Fixed = Mathematics::fixed<short, 11>;

		VCL_STRONG_INLINE VectorScalar() = default;
		VCL_STRONG_INLINE VectorScalar(Fixed s) { set(s); }
		VCL_STRONG_INLINE VectorScalar(Fixed s0, Fixed s1, Fixed s2, Fixed s3, Fixed s4, Fixed s5, Fixed s6, Fixed s7)
		{
			set(s0.data(), s1.data(), s2.data(), s3.data(), s4.data(), s5.data(), s6.data(), s7.data());
		}
		VCL_STRONG_INLINE explicit VectorScalar(const Base::RegType& v0)
		{
			set(v0);
		}

		VCL_STRONG_INLINE Base::RegType get(int i) const
		{
			VclRequire(0 <= i && i < NrRegs, "Access is in range.");
			return _data[i];
		}

	public:
		VCL_SIMD_BINARY_OP(operator+, _mm_add_epi16, 1)
		VCL_SIMD_BINARY_OP(operator-, _mm_sub_epi16, 1)
		//VCL_SIMD_BINARY_OP(operator*, _mmVCL_mullo_epi16, 1)
		
	public:
		VCL_SIMD_ASSIGN_OP(operator+=, _mm_add_epi16, 1)
		VCL_SIMD_ASSIGN_OP(operator-=, _mm_sub_epi16, 1)
		//VCL_SIMD_ASSIGN_OP(operator*=, _mmVCL_mullo_epi16, 1)
		
	public:
		//VCL_SIMD_COMP_OP(operator==, _mm_cmpeq_epi32,  1)
		//VCL_SIMD_COMP_OP(operator!=, _mm_cmpneq_epi32, 1)
		//VCL_SIMD_COMP_OP(operator<,  _mm_cmplt_epi32,  1)
		//VCL_SIMD_COMP_OP(operator<=, _mm_cmple_epi32,  1)
		//VCL_SIMD_COMP_OP(operator>,  _mm_cmpgt_epi32,  1)
		//VCL_SIMD_COMP_OP(operator>=, _mm_cmpge_epi32,  1)

	public:
		//VCL_SIMD_UNARY_OP(abs, Core::Simd::SSE::abs_s32, 1)
		
	public:
		//VCL_SIMD_BINARY_OP(operator&, _mm_and_si128, 1)
		//VCL_SIMD_BINARY_OP(operator|, _mm_or_si128, 1)

		//VCL_SIMD_BINARY_OP(min, Core::Simd::SSE::min_s32, 1)
		//VCL_SIMD_BINARY_OP(max, Core::Simd::SSE::max_s32, 1)
	};

	//VCL_STRONG_INLINE VectorScalar<int, 4> select(const VectorScalar<bool, 4>& mask, const VectorScalar<int, 4>& a, const VectorScalar<int, 4>& b)
	//{
	//	return VectorScalar<int, 4>(Core::Simd::SSE::blend_s32(b.get(0), a.get(0), mask.get(0)));
	//}
	//
	//VCL_STRONG_INLINE std::ostream& operator<< (std::ostream &s, const VectorScalar<int, 4>& rhs)
	//{
	//	alignas(16) int vars[4];
	//	_mm_store_si128(reinterpret_cast<__m128i*>(vars + 0), rhs.get(0));
	//
	//	s << "'" << vars[0] << ", " << vars[1] << ", " << vars[2] << ", " << vars[3] << "'";
	//
	//	return s;
	//}
}
