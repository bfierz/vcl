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

// C++ Standard Library
#include <string>
#include <type_traits>
#include <vector>

// Include the relevant parts from the library
#include <vcl/core/container/enumerate.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(EnumerateTest, EnumerateItems)
{
	using namespace Vcl::Core;

	std::vector<std::string> vec = { "First", "Second" };
	size_t count = 0;
	for (const auto& e : enumerate(vec))
	{
		EXPECT_EQ(count, e.first);
		EXPECT_EQ(vec[count], e.second);
		count++;
	}
	EXPECT_EQ(vec.size(), count);
}

TEST(EnumerateTest, EnumerateModifiesItems)
{
	using namespace Vcl::Core;

	std::vector<int> vec = { 1, 2, 3 };
	for (auto e : enumerate(vec))
		e.second += static_cast<int>(e.first);

	EXPECT_EQ((std::vector<int>{ 1, 3, 5 }), vec);
}

TEST(EnumerateTest, ConstEnumerateItems)
{
	using namespace Vcl::Core;

	const std::vector<std::string> vec = { "First", "Second" };
	size_t count = 0;
	for (const auto& e : const_enumerate(vec))
	{
		static_assert(std::is_same<decltype(e.second), const std::string&>::value, "Items must be const");
		EXPECT_EQ(count, e.first);
		EXPECT_EQ(vec[count], e.second);
		count++;
	}
	EXPECT_EQ(vec.size(), count);
}

TEST(EnumerateTest, EnumerateEmptyContainer)
{
	using namespace Vcl::Core;

	std::vector<int> vec;
	size_t count = 0;
	for (const auto& e : enumerate(vec))
	{
		(void)e;
		count++;
	}
	EXPECT_EQ(0u, count);
}
