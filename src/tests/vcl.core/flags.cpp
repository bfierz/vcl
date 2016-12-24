/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#include <vcl/core/flags.h>

// Google test
#include <gtest/gtest.h>

VCL_DECLARE_FLAGS(CarFeatures, Brackes, Engine, Seats)

TEST(FlagsTest, Construct)
{
	using namespace Vcl;

	Flags<CarFeatures> types{ CarFeatures::Engine, CarFeatures::Seats };

	EXPECT_TRUE(types.isAnySet());
	EXPECT_TRUE(types.isSet(CarFeatures::Engine));
	EXPECT_FALSE(types.isSet(CarFeatures::Brackes));
}

TEST(FlagsTest, Set)
{
	using namespace Vcl;

	Flags<CarFeatures> types;

	EXPECT_FALSE(types.isSet(CarFeatures::Seats));

	types.set(CarFeatures::Seats);
	EXPECT_TRUE(types.isSet(CarFeatures::Seats));

	types |= CarFeatures::Engine;
	EXPECT_TRUE(types.isSet(CarFeatures::Engine));

	auto types2 = types | CarFeatures::Brackes;
	EXPECT_TRUE(types2.isSet(CarFeatures::Brackes));
	EXPECT_TRUE(types2.areAllSet());
}

TEST(FlagsTest, Remove)
{
	using namespace Vcl;

	Flags<CarFeatures> types{ CarFeatures::Engine };

	EXPECT_TRUE(types.isSet(CarFeatures::Engine));

	types.remove(CarFeatures::Engine);

	EXPECT_FALSE(types.isSet(CarFeatures::Engine));
}
