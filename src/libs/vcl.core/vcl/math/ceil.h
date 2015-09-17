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

namespace Vcl { namespace Mathematics
{
	template<int N>
	inline unsigned int ceil(unsigned int val)
	{
		unsigned int div = (val + (N-1)) / N;
		return div * N;
	}

	template<>
	inline unsigned int ceil<8>(unsigned int val)
	{
		unsigned int res = (val +   7) & 0xfffffff8;
		return res;
	}

	template<>
	inline unsigned int ceil<16>(unsigned int val)
	{
		unsigned int res = (val +  15) & 0xfffffff0;
		return res;
	}

	template<>
	inline unsigned int ceil<32>(unsigned int val)
	{
		unsigned int res = (val +  31) & 0xffffffe0;
		return res;
	}

	template<>
	inline unsigned int ceil<64>(unsigned int val)
	{
		unsigned int res = (val +  63) & 0xffffffc0;
		return res;
	}

	template<>
	inline unsigned int ceil<128>(unsigned int val)
	{
		unsigned int res = (val + 127) & 0xffffff80;
		return res;
	}
	
	template<>
	inline unsigned int ceil<256>(unsigned int val)
	{
		unsigned int res = (val + 255) & 0xffffff00;
		return res;
	}
	
	template<>
	inline unsigned int ceil<512>(unsigned int val)
	{
		unsigned int res = (val + 511) & 0xfffffe00;
		return res;
	}

	template<>
	inline unsigned int ceil<1024>(unsigned int val)
	{
		unsigned int res = (val + 1023) & 0xfffffC00;
		return res;
	}
}}
