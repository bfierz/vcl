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

// Public implementations of the Fnv1a hashing function for C++:
// http://www.altdevblogaday.com/2011/10/27/quasi-compile-time-string-hashing/
// https://notes.underscorediscovery.com/constexpr-fnv1a/

namespace Vcl { namespace Util {
	// Source for prime-numbers:
	// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
	constexpr uint32_t fnv1a_32_offset = 0x811c9dc5;
	constexpr uint32_t fnv1a_32_prime = 0x01000193;
	constexpr uint64_t fnv1a_64_offset = 0xcbf29ce484222325;
	constexpr uint64_t fnv1a_64_prime = 0x100000001b3;

	VCL_STRONG_INLINE constexpr uint32_t calculateFnv1a32(const char* str) noexcept
	{
		uint32_t hash = fnv1a_32_offset;

		while (*str != 0)
		{
			hash ^= uint32_t(*str++);
			hash *= fnv1a_32_prime;
		}

		return hash;
	}

	VCL_STRONG_INLINE constexpr uint32_t calculateFnv1a32(const char* str, size_t length) noexcept
	{
		uint32_t hash = fnv1a_32_offset;

		for (size_t i = 0; i < length; ++i)
		{
			hash ^= uint32_t(*str++);
			hash *= fnv1a_32_prime;
		}

		return hash;
	}

	VCL_STRONG_INLINE constexpr uint64_t calculateFnv1a64(const char* str) noexcept
	{
		uint64_t hash = fnv1a_64_offset;

		while (*str != 0)
		{
			hash ^= uint32_t(*str++);
			hash *= fnv1a_64_prime;
		}

		return hash;
	}

	VCL_STRONG_INLINE constexpr uint64_t calculateFnv1a64(const char* str, size_t length) noexcept
	{
		uint64_t hash = fnv1a_64_offset;

		for (size_t i = 0; i < length; ++i)
		{
			hash ^= uint32_t(*str++);
			hash *= fnv1a_64_prime;
		}

		return hash;
	}

	template<unsigned int N, unsigned int I>
	struct FnvHash
	{
		VCL_STRONG_INLINE static unsigned int hash(const char (&str)[N])
		{
			return (FnvHash<N, I - 1>::hash(str) ^ static_cast<unsigned int>(str[I - 2])) * fnv1a_32_prime;
		}
	};

	template<unsigned int N>
	struct FnvHash<N, 1>
	{
		VCL_STRONG_INLINE static unsigned int hash(const char (&str)[N])
		{
			VCL_UNREFERENCED_PARAMETER(str);
			return fnv1a_32_offset;
		}
	};

	class StringHash
	{
	public:
		struct DynamicConstCharString
		{
			VCL_STRONG_INLINE DynamicConstCharString(const char* s)
			: str(s) {}
			const char* str;
		};

	public:
		template<size_t N>
		VCL_STRONG_INLINE constexpr StringHash(const char (&str)[N])
		: _hash(FnvHash<N, N>::hash(str))
		{
		}

		VCL_STRONG_INLINE StringHash(DynamicConstCharString str)
		: _hash(calculateFnv1a32(str.str))
		{
		}

		constexpr VCL_STRONG_INLINE explicit StringHash(const char* s, size_t length)
		: _hash(calculateFnv1a32(s, length))
		{
		}

		constexpr uint32_t hash() const
		{
			return _hash;
		}

	private:
		//! Computed hash value
		uint32_t _hash;
	};

	namespace Literals {
		constexpr uint32_t operator"" _fnv1a32(const char* str, size_t N)
		{
			return calculateFnv1a32(str, N);
		}
		constexpr uint64_t operator"" _fnv1a64(const char* str, size_t N)
		{
			return calculateFnv1a64(str, N);
		}
	}
}}
