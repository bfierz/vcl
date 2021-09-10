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

namespace Vcl { namespace Mathematics {
	inline constexpr uint32_t ceil(uint32_t val, uint32_t N) noexcept
	{
		const uint32_t div = (val + (N - 1)) / N;
		return div * N;
	}

	template<int N>
	inline uint32_t ceil(uint32_t val) noexcept
	{
		const uint32_t div = (val + (N - 1)) / N;
		return div * N;
	}

	template<>
	inline uint32_t ceil<8>(uint32_t val) noexcept
	{
		const uint32_t res = (val + 7) & 0xfffffff8;
		return res;
	}

	template<>
	inline uint32_t ceil<16>(uint32_t val) noexcept
	{
		const uint32_t res = (val + 15) & 0xfffffff0;
		return res;
	}

	template<>
	inline uint32_t ceil<32>(uint32_t val) noexcept
	{
		const uint32_t res = (val + 31) & 0xffffffe0;
		return res;
	}

	template<>
	inline uint32_t ceil<64>(uint32_t val) noexcept
	{
		const uint32_t res = (val + 63) & 0xffffffc0;
		return res;
	}

	template<>
	inline uint32_t ceil<128>(uint32_t val) noexcept
	{
		const uint32_t res = (val + 127) & 0xffffff80;
		return res;
	}

	template<>
	inline uint32_t ceil<256>(uint32_t val) noexcept
	{
		const uint32_t res = (val + 255) & 0xffffff00;
		return res;
	}

	template<>
	inline uint32_t ceil<512>(uint32_t val) noexcept
	{
		const uint32_t res = (val + 511) & 0xfffffe00;
		return res;
	}

	template<>
	inline uint32_t ceil<1024>(uint32_t val) noexcept
	{
		const uint32_t res = (val + 1023) & 0xfffffC00;
		return res;
	}

	inline constexpr uint64_t ceil(uint64_t val, uint64_t N) noexcept
	{
		const uint64_t div = (val + (N - 1)) / N;
		return div * N;
	}

	template<int N>
	inline uint64_t ceil(uint64_t val) noexcept
	{
		const uint64_t div = (val + (N - 1)) / N;
		return div * N;
	}

	template<>
	inline uint64_t ceil<8>(uint64_t val) noexcept
	{
		const uint64_t res = (val + 7) & 0xfffffffffffffff8;
		return res;
	}

	template<>
	inline uint64_t ceil<16>(uint64_t val) noexcept
	{
		const uint64_t res = (val + 15) & 0xfffffffffffffff0;
		return res;
	}

	template<>
	inline uint64_t ceil<32>(uint64_t val) noexcept
	{
		const uint64_t res = (val + 31) & 0xffffffffffffffe0;
		return res;
	}

	template<>
	inline uint64_t ceil<64>(uint64_t val) noexcept
	{
		const uint64_t res = (val + 63) & 0xffffffffffffffc0;
		return res;
	}

	template<>
	inline uint64_t ceil<128>(uint64_t val) noexcept
	{
		const uint64_t res = (val + 127) & 0xffffffffffffff80;
		return res;
	}

	template<>
	inline uint64_t ceil<256>(uint64_t val) noexcept
	{
		const uint64_t res = (val + 255) & 0xffffffffffffff00;
		return res;
	}

	template<>
	inline uint64_t ceil<512>(uint64_t val) noexcept
	{
		const uint64_t res = (val + 511) & 0xfffffffffffffe00;
		return res;
	}

	template<>
	inline uint64_t ceil<1024>(uint64_t val) noexcept
	{
		const uint64_t res = (val + 1023) & 0xfffffffffffffC00;
		return res;
	}
}}
