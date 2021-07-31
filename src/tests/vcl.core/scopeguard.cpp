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
#include <vcl/util/scopeguard.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(ScopeGuardTest, ScopeGuardExit)
{
	bool guard_triggered = false;

	{
		VCL_SCOPE_EXIT { guard_triggered = true; };
	}

	EXPECT_TRUE(guard_triggered) << "Exit guard triggered.";
}

TEST(ScopeGuardTest, ScopeGuardFail)
{
	bool guard_triggered = false;

	try
	{
		VCL_SCOPE_FAIL { guard_triggered = true; };
		{
			throw std::exception{};
		}
	} catch (...)
	{
	}

	EXPECT_TRUE(guard_triggered) << "Failure guard triggered.";
}

TEST(ScopeGuardTest, ScopeGuardSuccess)
{
	bool guard_triggered = false;

	{
		VCL_SCOPE_SUCCESS { guard_triggered = true; };
	}

	EXPECT_TRUE(guard_triggered) << "Success guard triggered.";
}
