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

// VCL
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl { namespace Util {
	template<int N>
	struct FastMath
	{
		template<typename T>
		static T modulo(const T& x) noexcept
		{
			const int m = x % static_cast<T>(N);
			return select(m < 0, m + T(N), m);
		}
	};
	template<>
	struct FastMath<16>
	{
		template<typename T>
		static T modulo(const T& x) noexcept
		{
			return x & static_cast<T>(15);
		}
	};
	template<>
	struct FastMath<32>
	{
		template<typename T>
		static T modulo(const T& x) noexcept
		{
			return x & static_cast<T>(31);
		}
	};
	template<>
	struct FastMath<64>
	{
		template<typename T>
		static T modulo(const T& x) noexcept
		{
			return x & static_cast<T>(63);
		}
	};
	template<>
	struct FastMath<128>
	{
		template<typename T>
		static T modulo(const T& x) noexcept
		{
			return x & static_cast<T>(127);
		}
	};
}}
