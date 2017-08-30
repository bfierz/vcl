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
#include <vcl/core/memory/smart_ptr.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(OwnerPtrTest, Simple)
{
	using namespace Vcl::Core;

	int i = 5;

	// Owner and ref in same scope
	{
		owner_ptr<int> owner{ &i };
		ref_ptr<int> ref{ owner };

		EXPECT_EQ(owner.get(), ref.get());
		EXPECT_TRUE(ref);
		EXPECT_EQ(1, owner.use_count());

		ref_ptr<int> ref2{ owner };
		EXPECT_EQ(2, owner.use_count());
	}

	// Owner and ref in different scopes
	{
		ref_ptr<int> ref;
		{
			owner_ptr<int> owner{ &i };
			ref.reset(owner);

			EXPECT_EQ(owner.get(), ref.get());
			EXPECT_TRUE(ref);
			EXPECT_EQ(1, owner.use_count());
		}
		
		EXPECT_TRUE(ref);
	}
}
