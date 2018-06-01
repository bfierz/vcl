/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <vcl/core/convert.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(ConvertTest, String)
{
	using namespace Vcl;

	EXPECT_EQ(std::string{ "Test" }, to_string(std::string{ "Test" }));
	EXPECT_EQ(std::string{ "Test" }, from_string<std::string>(std::string{ "Test" }));
}

TEST(ConvertTest, Bool)
{
	using namespace Vcl;

	EXPECT_EQ(std::string{ "true" }, to_string(true));
	EXPECT_EQ(std::string{ "false" }, to_string(false));
	EXPECT_EQ(true, from_string<bool>(std::string{ "true" }));
	EXPECT_EQ(true, from_string<bool>(std::string{ "1" }));
	EXPECT_EQ(false, from_string<bool>(std::string{ "false" }));
	EXPECT_EQ(false, from_string<bool>(std::string{ "0" }));
}

TEST(ConvertTest, Float)
{
	using namespace Vcl;

	EXPECT_EQ(0, to_string(4.67f).find(std::string{ "4.67" }));
	EXPECT_EQ(0, to_string(-4.67f).find(std::string{ "-4.67" }));
	EXPECT_EQ(4.67f, from_string<float>(std::string{ "4.67f" }));
	EXPECT_EQ(-4.67f, from_string<float>(std::string{ "-4.67f" }));
}

TEST(ConvertTest, Int)
{
	using namespace Vcl;

	EXPECT_EQ(std::string{ "4" }, to_string(4));
	EXPECT_EQ(std::string{ "-4" }, to_string(-4));
	EXPECT_EQ(4, from_string<float>(std::string{ "4" }));
	EXPECT_EQ(-4, from_string<float>(std::string{ "-4" }));
}
