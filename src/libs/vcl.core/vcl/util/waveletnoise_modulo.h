/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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

// C++ standard library
#include <cstdint>

namespace Vcl { namespace Util
{
	template<int N>
	VCL_CPP_CONSTEXPR_14 inline int fast_modulo(int x)
	{
		const int m = x % N;
		return (m < 0) ? m + N : m;
	}
	template<> VCL_CPP_CONSTEXPR_14 inline int fast_modulo<128>(int x) { return x & 127; }
	template<> VCL_CPP_CONSTEXPR_14 inline int fast_modulo< 64>(int x) { return x &  63; }
	template<> VCL_CPP_CONSTEXPR_14 inline int fast_modulo< 32>(int x) { return x &  31; }
	template<> VCL_CPP_CONSTEXPR_14 inline int fast_modulo< 16>(int x) { return x &  15; }

#if PTRDIFF_MAX != INT32_MAX
	template<int N>
	VCL_CPP_CONSTEXPR_14 inline ptrdiff_t fast_modulo(ptrdiff_t x)
	{
		const ptrdiff_t m = x % N;
		return (m < 0) ? m + N : m;
	}
	template<> VCL_CPP_CONSTEXPR_14 inline ptrdiff_t fast_modulo<128>(ptrdiff_t x) { return x & 127; }
	template<> VCL_CPP_CONSTEXPR_14 inline ptrdiff_t fast_modulo< 64>(ptrdiff_t x) { return x & 63; }
	template<> VCL_CPP_CONSTEXPR_14 inline ptrdiff_t fast_modulo< 32>(ptrdiff_t x) { return x & 31; }
	template<> VCL_CPP_CONSTEXPR_14 inline ptrdiff_t fast_modulo< 16>(ptrdiff_t x) { return x & 15; }
#endif
}}
