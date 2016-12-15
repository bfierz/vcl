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

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/commandqueue.h>
#include <vcl/compute/cuda/context.h>
#include <vcl/compute/cuda/device.h>
#include <vcl/compute/cuda/platform.h>
#include <vcl/core/memory/smart_ptr.h>

// Google test
#include <gtest/gtest.h>

TEST(Cuda, QueryDevices)
{
	using namespace Vcl::Compute::Cuda;

	Platform::initialise();
	Platform* platform = Platform::instance();
	const Device& device = platform->device(0);
	auto ctx = Vcl::make_owner<Context>(device);
	ctx->bind();

	CUcontext curr{ 0 };
	cuCtxPopCurrent(&curr);

	EXPECT_EQ((CUcontext) *ctx, curr) << "Context was not set correctly";
	
	Platform::dispose();
}

TEST(Cuda, InitBuffer)
{
	using namespace Vcl::Compute::Cuda;

	Platform::initialise();
	Platform* platform = Platform::instance();
	const Device& device = platform->device(0);
	auto ctx = Vcl::make_owner<Context>(device);
	ctx->bind();

	auto queue = ctx->createCommandQueue();
	auto buffer = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 128);

	unsigned int pattern = 0xdeadbeef;
	queue->fill(buffer, &pattern, sizeof(pattern));

	std::vector<unsigned int> result(32);
	queue->read(result.data(), buffer, true);

	for (unsigned int p : result)
		EXPECT_EQ(pattern, p) << "Pattern was not set correctly";

	Platform::dispose();
}

TEST(Cuda, InitUnifiedMemory)
{
	using namespace Vcl::Compute::Cuda;

	Platform::initialise();
	Platform* platform = Platform::instance();
	const Device& device = platform->device(0);
	auto ctx = Vcl::make_owner<Context>(device);
	ctx->bind();

	auto queue = ctx->createCommandQueue();
	auto buffer = ctx->createBuffer(Vcl::Compute::BufferAccess::Unified, 128);

	unsigned int pattern = 0xdeadbeef;
	queue->fill(buffer, &pattern, sizeof(pattern));
	queue->sync();
	ctx->sync();

	std::vector<unsigned int> result(32);
	auto src_ptr = (const void*)(CUdeviceptr)*Vcl::static_pointer_cast<Vcl::Compute::Cuda::Buffer>(buffer);
	memcpy(result.data(), src_ptr, 32*sizeof(unsigned int));

	for (unsigned int p : result)
		EXPECT_EQ(pattern, p) << "Pattern was not set correctly";

	Platform::dispose();
}

