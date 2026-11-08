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

// VCL configuration
#include <vcl/config/global.h>

// Include the relevant parts from the library
#include <vcl/core/enummap.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

namespace {
	// Starts at one and has a gap, like the graphics state enumerations
	enum class Colour
	{
		Red = 1,
		Green = 2,
		Blue = 5,
	};

	// Keys deliberately not in declaration order
	constexpr Vcl::EnumMapping<Colour, int> ColourTable[] = {
		{ Colour::Blue, 0x0000ff },
		{ Colour::Red, 0xff0000 },
		{ Colour::Green, 0x00ff00 },
	};
	constexpr Vcl::EnumMap<Colour, int, Vcl::enumMapSize(ColourTable)> Colours{ ColourTable };

	static_assert(Vcl::enumMapSize(ColourTable) == 6, "Size is derived from the largest key.");
	static_assert(Colours.contains(Colour::Blue), "Mapped keys are found at compile time.");
	static_assert(!Colours.contains(static_cast<Colour>(3)), "Gaps are not mapped.");
	static_assert(!Colours.contains(static_cast<Colour>(42)), "Keys out of range are not mapped.");
}

TEST(EnumMapTest, Lookup)
{
	EXPECT_EQ(0xff0000, Colours[Colour::Red]);
	EXPECT_EQ(0x00ff00, Colours[Colour::Green]);
	EXPECT_EQ(0x0000ff, Colours[Colour::Blue]);
}

TEST(EnumMapTest, Contains)
{
	EXPECT_TRUE(Colours.contains(Colour::Red));
	EXPECT_FALSE(Colours.contains(static_cast<Colour>(0)));
	EXPECT_FALSE(Colours.contains(static_cast<Colour>(4)));
	EXPECT_FALSE(Colours.contains(static_cast<Colour>(6)));
}
