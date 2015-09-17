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

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Util
{
	/*!
	 * Original implementation can be found at: http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
	 */
	class MortonCode
	{
	public:
		VCL_STRONG_INLINE static uint64_t encode(uint32_t x, uint32_t y, uint32_t z)
		{
			uint64_t answer = 0;
			answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
			return answer;
		}

		VCL_STRONG_INLINE static void decode(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z)
		{
			x = 0;
			y = 0;
			z = 0;

			for (uint64_t i = 0; i < (sizeof(uint64_t) * CHAR_BIT) / 3; ++i)
			{
				x |= ((morton & (uint64_t(1ull) << uint64_t((3ull * i) + 0ull))) >> uint64_t(((3ull * i) + 0ull) - i));
				y |= ((morton & (uint64_t(1ull) << uint64_t((3ull * i) + 1ull))) >> uint64_t(((3ull * i) + 1ull) - i));
				z |= ((morton & (uint64_t(1ull) << uint64_t((3ull * i) + 2ull))) >> uint64_t(((3ull * i) + 2ull) - i));
			}
		}

	private:
		VCL_STRONG_INLINE static uint64_t splitBy3(uint32_t a)
		{
			Require((a & ~0x1fffff) == 0, "a is in only 21 bits large.");

			uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
			x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
			x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
			x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
			x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
			x = (x | x << 2) & 0x1249249249249249;
			return x;
		}
	};
}}
