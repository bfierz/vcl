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

// VCL configuration
#include <vcl/config/global.h>

// Include the relevant parts from the library
#include <vcl/core/container/bitvector.h>

// C++ standard library
#include <random>
#include <vector>

VCL_BEGIN_EXTERNAL_HEADERS

// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

using TempBitVector = Vcl::Core::BitVector<>;

TEST(BitVectorTest, SetBit)
{
	using namespace Vcl::Core;

	TempBitVector v;
	v.assign(19, false);

	EXPECT_FALSE(v[17]) << "Element 17 is false";

	v[17] = true;

	EXPECT_TRUE(v[17]) << "Element 17 is true";
}

TEST(BitVectorTest, Clear)
{
	using namespace Vcl::Core;

	TempBitVector v;
	v.assign(19, false);

	EXPECT_EQ(19, v.size()) << "Size is correct";

	v.clear();

	EXPECT_EQ(0, v.size()) << "Vector is empty";
}

TEST(BitVectorTest, ResizeWithValue)
{
	using namespace Vcl::Core;

	TempBitVector v;
	v.assign(19, true);

	EXPECT_TRUE(v[17]) << "Element 17 is true";

	v[17] = false;

	EXPECT_FALSE(v[17]) << "Element 17 is false";
}

TEST(BitVectorTest, Generation)
{
	using namespace Vcl::Core;

	TempBitVector v(19, false);
	EXPECT_EQ(v.generation(), 1);
	EXPECT_FALSE((bool)v[17]) << "Element 17 is false";

	for (int i = 1; i < std::numeric_limits<TempBitVector::container_t::value_type>::max(); i++)
	{
		v[17] = true;
		EXPECT_TRUE(v[17]) << "Element 17 is true";
		v.assign(19, false);
		EXPECT_FALSE(v[17]) << "Element 17 is false";
		EXPECT_EQ(v.generation(), i + 1);
	}
	v.assign(19, false);
	EXPECT_EQ(v.generation(), 1);
}
