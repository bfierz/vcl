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

// Abseil
#include <absl/types/span.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Util {
	/*!
	 * Original implementation can be found at:
	 * http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
	 * Updated version found here:
	 * https://github.com/Forceflow/libmorton
	 *
	 * Refactor to use library
	 */
	class MortonCode
	{
	public:
		VCL_STRONG_INLINE static uint64_t encode(uint32_t x, uint32_t y, uint32_t z)
		{
			uint64_t answer = 0;
			answer |= splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);

			VclAssertBlock
			{
				uint32_t x2, y2, z2;
				decode(answer, x2, y2, z2);
				VclEnsure(x == x2 && y == y2 && z == z2, "Encoding is consistent");
			}
			return answer;
		}

		VCL_STRONG_INLINE static void decode(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z)
		{
			x = 0;
			y = 0;
			z = 0;

			for (uint64_t i = 0; i < (sizeof(uint64_t) * CHAR_BIT) / 3; ++i)
			{
				x |= (morton & (1ull << (3ull * i + 0ull))) >> (2ull * i + 0ull);
				y |= (morton & (1ull << (3ull * i + 1ull))) >> (2ull * i + 1ull);
				z |= (morton & (1ull << (3ull * i + 2ull))) >> (2ull * i + 2ull);
			}
		}

		VCL_STRONG_INLINE static Eigen::AlignedBox3f calculateBoundingBox
		(
			uint64_t morton,
			float cell_size
		)
		{
			uint32_t x, y, z;
			decode(morton, x, y, z);

			return{
				Eigen::Vector3f(x+0, y+0, z+0)*cell_size,
				Eigen::Vector3f(x+1, y+1, z+1)*cell_size
			};
		}

	private:
		VCL_STRONG_INLINE static uint64_t splitBy3(uint32_t a)
		{
			VclRequire((a & ~0x1fffff) == 0, "a is in only 21 bits large.");

			uint64_t x = a & 0x1fffff;             // we only look at the first 21 bits
			x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
			x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
			x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
			x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
			x = (x | x << 2) & 0x1249249249249249;
			return x;
		}
	};

	//! Split the list of sorted morton codes at the highest differing bit
	//! \param sorted_morton_codes List of sorted morton codes to split
	//! This implementations is adapted from Tero Karras version published in
	//! http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
	template<typename T>
	int splitHighestBit
	(
		absl::Span<const std::pair<uint64_t, T>> sorted_morton_codes,
		int first, int last
	)
	{
		VclRequire(std::is_sorted(
			std::begin(sorted_morton_codes), 
			std::end(sorted_morton_codes), 
			[](const auto& a, const auto& b) { return a.first < b.first; }),
			"Morton codes are sorted");

		using Vcl::Mathematics::clz64;

		// Identical Morton codes => split the range in the middle.
		uint64_t first_code = sorted_morton_codes[first].first;
		uint64_t last_code = sorted_morton_codes[last].first;
		if (first_code == last_code)
			return (first + last) >> 1;

		// Calculate the number of highest bits that are the same
		// for all objects, using the count-leading-zeros intrinsic.
		const auto common_prefix = clz64(first_code ^ last_code);

		// Use binary search to find where the next bit differs.
		// Specifically, we are looking for the highest object that
		// shares more than common_prefix bits with the first one.
		size_t split = first; // Initial guess
		size_t step = last - first;
		do
		{
			step = (step + 1) >> 1; // Exponential decrease
			int new_split = split + step; // Proposed new position

			if (new_split < last)
			{
				const auto split_code = sorted_morton_codes[new_split].first;
				const auto split_prefix = clz64(first_code ^ split_code);
				if (split_prefix > common_prefix)
					split = new_split;
			}
		} while (step > 1);

		return split;
	}
}}
