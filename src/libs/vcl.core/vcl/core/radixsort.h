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
#include <algorithm>
#include <vector>

namespace Vcl
{
	/*!
	 * Sorts input arrays of unsigned integer keys
	 * 
	 * \param keys        Array of keys for data to be sorted
	 * \param keyBits     The number of bits in each key to use for ordering
	 *
	 * \note This implementation is base on http://rosettacode.org/wiki/Sorting_algorithms/Radix_sort#C.2B.2B
	 */
	template<typename BidirIt>
	void radix_sort
	(
		BidirIt first,
		BidirIt last,
		unsigned int keyBits
	)
	{
		for (int lsb = 0; lsb < keyBits; ++lsb) // least-significant-bit
		{
			std::stable_partition(keys.data(), keys.data() + keys.size(), [lsb] (int value)
			{
				if (lsb == 31) // sign bit
					return value < 0; // negative int to left partition
				else
					return !(value & (1 << lsb)); // 0 bit to left partition
			});
		}
	}
}
