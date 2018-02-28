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
#include <vcl/config/eigen.h>

// C++ standard library
#include <initializer_list>

// VCL
#include <vcl/core/preprocessor.h>

// Based on http://molecularmusings.wordpress.com/2011/08/23/flags-on-steroids/

#define VCL_DECLARE_FLAGS_ENUM(name, n)                    name = (1u << n),
#define VCL_DECLARE_FLAGS_BITS(name, n)                    uint32_t name : 1;
#define VCL_DECLARE_FLAGS_TO_STRING(name, n)               case name: return VCL_PP_STRINGIZE(name);

#define VCL_DECLARE_FLAGS(name, ...)                                     \
struct name		                                                         \
{                                                                        \
	static const size_t Count = VCL_PP_VA_NUM_ARGS(__VA_ARGS__);         \
	static_assert(Count <= sizeof(uint32_t)*8, "Too many flags");        \
	enum Enum                                                            \
	{                                                                    \
		VCL_PP_VA_EXPAND_ARGS (VCL_DECLARE_FLAGS_ENUM, __VA_ARGS__)      \
	};                                                                   \
	struct Bits                                                          \
	{                                                                    \
		VCL_PP_VA_EXPAND_ARGS (VCL_DECLARE_FLAGS_BITS, __VA_ARGS__)      \
	};                                                                   \
	static const char* ToString(size_t value)                            \
	{                                                                    \
		switch (value)                                                   \
		{                                                                \
		VCL_PP_VA_EXPAND_ARGS (VCL_DECLARE_FLAGS_TO_STRING, __VA_ARGS__) \
		default:                                                         \
			VCL_NO_SWITCH_DEFAULT;                                       \
		}                                                                \
		return "";                                                       \
	}                                                                    \
};																		 \
inline Vcl::Flags<name> operator|(name::Enum lhs, name::Enum rhs)        \
{                                                                        \
	return (Vcl::Flags<name>(lhs) | Vcl::Flags<name>(rhs));              \
}																		 \
inline Vcl::Flags<name> operator|(Vcl::Flags<name> lhs, name::Enum rhs)  \
{                                                                        \
	return (lhs | Vcl::Flags<name>(rhs));								 \
}																		 \
inline Vcl::Flags<name> operator|(name::Enum lhs, Vcl::Flags<name> rhs)  \
{                                                                        \
	return (Vcl::Flags<name>(lhs) | rhs);						         \
}

namespace Vcl
{
	template <class T>
	class Flags
	{
	private:
		typedef typename T::Enum Enum;
		typedef typename T::Bits Bits;

	public:
		inline Flags(void)
		: _flags(0)
		{
		}

		inline Flags(Enum flag)
		: _flags(flag)
		{
		}

		inline Flags(std::initializer_list<Enum> flags)
		: _flags(0)
		{
			for (const auto flag : flags)
				_flags |= flag;
		}

		inline void set(Enum flag)
		{
			_flags |= flag;
		}

		inline void remove(Enum flag)
		{
			_flags &= ~flag;
		}

		inline void clear(void)
		{
			_flags = 0;
		}

		inline bool isSet(Enum flag) const
		{
			return ((_flags & flag) != 0);
		}

		inline bool isAnySet(void) const
		{
			return (_flags != 0);
		}

		inline bool areAllSet(void) const
		{
			return (_flags == ((1ull << T::Count) - 1u));
		}

	public: // Operators
		inline Flags operator|(Flags other) const
		{
			return Flags(_flags | other._flags);
		}

		inline Flags& operator|=(Flags other)
		{
			_flags |= other._flags;
			return *this;
		}

	private:
		inline explicit Flags(uint32_t flags)
		: _flags(flags)
		{
		}

		union
		{
			uint32_t _flags;
			Bits _bits;
		};
	};
}
