/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2026 Basil Fierz
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
#include <cstddef>
#include <type_traits>

// VCL
#include <vcl/core/contract.h>

namespace Vcl {
	/*!
	 * \brief One entry of a mapping table from an enumeration to another value
	 */
	template<typename From, typename To>
	struct EnumMapping
	{
		From Key;
		To Value;
	};

	namespace Detail {
		template<typename Enum>
		constexpr size_t enumIndex(Enum e)
		{
			return static_cast<size_t>(static_cast<std::underlying_type_t<Enum>>(e));
		}

		//! Not constexpr on purpose: reaching it during constant evaluation is a compile error
		inline void invalidEnumMapTable() {}
	}

	//! Number of slots needed to index a table by its largest key
	template<typename From, typename To, size_t N>
	constexpr size_t enumMapSize(const EnumMapping<From, To> (&table)[N])
	{
		size_t size = 0;
		for (const auto& entry : table)
		{
			if (static_cast<std::underlying_type_t<From>>(entry.Key) < 0)
				::Vcl::Detail::invalidEnumMapTable(); // Negative keys are not supported

			const size_t idx = ::Vcl::Detail::enumIndex(entry.Key) + 1;
			size = idx > size ? idx : size;
		}
		return size;
	}

	/*!
	 * \brief Constant-time conversion from an enumeration to another value
	 *
	 * The map is built at compile time from a list of key/value pairs. The keys
	 * may be listed in any order and the enumeration may have gaps. A lookup is
	 * a single array access; only debug builds check that the key is mapped.
	 * Unmapped slots hold a value-initialised \p To. Listing a key twice is a
	 * compile error.
	 *
	 * \code
	 * enum class Colour { Red = 1, Green = 2, Blue = 4 };
	 *
	 * constexpr EnumMapping<Colour, uint32_t> RgbTable[] = {
	 *     { Colour::Red,   0xff0000 },
	 *     { Colour::Green, 0x00ff00 },
	 *     { Colour::Blue,  0x0000ff },
	 * };
	 * constexpr EnumMap<Colour, uint32_t, enumMapSize(RgbTable)> Rgb{ RgbTable };
	 *
	 * uint32_t red = Rgb[Colour::Red]; // 0xff0000
	 * \endcode
	 *
	 * In a header, declare the table and the map as \c static \c constexpr locals
	 * of an inline function, so that all translation units share one copy.
	 */
	template<typename From, typename To, size_t Size>
	class EnumMap
	{
		static_assert(std::is_enum<From>::value, "Keys must be an enumeration.");

	public:
		template<size_t N>
		constexpr explicit EnumMap(const EnumMapping<From, To> (&table)[N])
		: _values{}, _mapped{}
		{
			for (const auto& entry : table)
			{
				const size_t idx = ::Vcl::Detail::enumIndex(entry.Key);
				if (idx >= Size || _mapped[idx])
					::Vcl::Detail::invalidEnumMapTable(); // Key out of range or listed twice

				_values[idx] = entry.Value;
				_mapped[idx] = true;
			}
		}

		To operator[](From key) const
		{
			const size_t idx = ::Vcl::Detail::enumIndex(key);
			VclRequire(idx < Size && _mapped[idx], "Enumeration value is valid.");
			return _values[idx];
		}

		constexpr bool contains(From key) const
		{
			const size_t idx = ::Vcl::Detail::enumIndex(key);
			return idx < Size && _mapped[idx];
		}

	private:
		To _values[Size];
		bool _mapped[Size];
	};
}
