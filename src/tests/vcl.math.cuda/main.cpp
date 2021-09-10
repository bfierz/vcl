/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2019 Basil Fierz
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
#include <gtest/gtest.h>

// VCL
#include <vcl/compute/cuda/context.h>
#include <vcl/compute/cuda/platform.h>
#include <vcl/core/memory/smart_ptr.h>

Vcl::owner_ptr<Vcl::Compute::Cuda::Context> default_ctx;

int main(int argc, char** argv)
{
	using namespace Vcl::Compute::Cuda;

	Platform::initialise();

	Platform* platform = Platform::instance();
	const Device& device = platform->device(0);
	default_ctx = Vcl::make_owner<Context>(device);
	default_ctx->bind();

	// Run the tests
	::testing::InitGoogleTest(&argc, argv);
	const int ret_code = RUN_ALL_TESTS();

	default_ctx.reset();

	Platform::dispose();

	return ret_code;
}
