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

// VCL configuration
#include <vcl/config/global.h>

// Include the relevant parts from the library
#include <vcl/core/memory/allocator.h>

// C++ standard library
#include <scoped_allocator>
#include <vector>

// Google test
#include <gtest/gtest.h>

struct SimpleObject
{
	SimpleObject()
	{
		x = 5;
	}

	SimpleObject(int v)
	{
		x = v;
	}

	SimpleObject(const SimpleObject& rhs)
	{
		x = rhs.x;
	}

	SimpleObject(SimpleObject&& rhs)
	{
		x = rhs.x;
	}

	int x;
};

TEST(AllocatorTest, StandardAllocInitObject)
{
	using namespace Vcl::Core;

	std::vector<SimpleObject, Allocator<SimpleObject, StandardAllocPolicy<SimpleObject>>> v(10, { 4 });

	// Check that copy constructor was used to initialized objects
	EXPECT_EQ(4, v[5].x);

	v.resize(15);

	// Check that default constructor was used to initialized objects
	EXPECT_EQ(5, v[13].x);
}

TEST(AllocatorTest, AlignedAllocInitObject)
{
	using namespace Vcl::Core;

	std::vector<SimpleObject, Allocator<SimpleObject, AlignedAllocPolicy<SimpleObject, 64>>> v(10, { 4 });

	// Check that copy constructor was used to initialized objects
	EXPECT_EQ(4, v[5].x);

	v.resize(15);

	// Check that default constructor was used to initialized objects
	EXPECT_EQ(5, v[13].x);

	// Check alignment of vector data
	auto base_ptr = reinterpret_cast<size_t>(v.data()) & 0x3f;
	EXPECT_EQ(0u, base_ptr);
}
