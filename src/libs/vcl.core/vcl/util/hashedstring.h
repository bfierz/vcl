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

// C++ standard libary
#include <cstring>
#include <limits>

namespace Vcl { namespace Util
{
	// The implementation the following methods is based on:
	// http://www.altdevblogaday.com/2011/10/27/quasi-compile-time-string-hashing/

	VCL_STRONG_INLINE unsigned int calculateFNV(const char* str)
	{
		const size_t length = strlen(str) + 1;
		unsigned int hash = 2166136261u;

		for (size_t i = 0; i < length; ++i)
		{
			hash ^= *str++;
			hash *= 16777619u;
		}
 
		return hash;
	}

	template <unsigned int N, unsigned int I>
	struct FnvHash
	{
		VCL_STRONG_INLINE static unsigned int hash(const char (&str)[N])
		{
			return (FnvHash<N, I-1>::hash(str) ^ str[I-1])*16777619u;
		}
	};
 
	template <unsigned int N>
	struct FnvHash<N, 1>
	{
		VCL_STRONG_INLINE static unsigned int hash(const char (&str)[N])
		{
			return (2166136261u ^ str[0])*16777619u;
		}
	};

	class StringHash
	{ 
	public:
		template <size_t N>
		VCL_STRONG_INLINE StringHash(const char (&str)[N])
		: mHash(FnvHash<N, N>::hash(str))
		{
		}
 
		VCL_STRONG_INLINE StringHash(const gsl::cstring_span<> str)
		: mHash(calculateFNV(str.data()))
		{
		}

		size_t hash() const
		{
			return mHash;
		}
 
	private:
		size_t mHash;
	};
}}
