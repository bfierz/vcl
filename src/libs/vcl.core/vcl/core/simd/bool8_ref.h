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

namespace Vcl
{
#ifdef VCL_COMPILER_MSVC
#	pragma warning(disable : 4324)
#endif
	template<>
	class alignas(16) VectorScalar<bool, 8> : protected Core::Simd::VectorScalarBase<bool, 8, Core::Simd::SimdExt::None>
	{
	public:
		VCL_SIMD_VECTORSCALAR_SETUP(None)
		VCL_STRONG_INLINE Scalar& operator[](int idx) noexcept { return _data[idx]; }

	public:
		VCL_SIMD_BINARY_OP(operator&&, Core::Simd::Details::conj, 8)
		VCL_SIMD_BINARY_OP(operator||, Core::Simd::Details::disj, 8)

		VCL_SIMD_ASSIGN_OP(operator&=, Core::Simd::Details::conj, 8)
		VCL_SIMD_ASSIGN_OP(operator|=, Core::Simd::Details::disj, 8)
	};
#ifdef VCL_COMPILER_MSVC
#	pragma warning(default : 4324)
#endif
}
