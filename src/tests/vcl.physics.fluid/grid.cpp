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
#include <vcl/compute/cuda/commandqueue.h>
#include <vcl/compute/cuda/context.h>
#include <vcl/compute/cuda/device.h>
#include <vcl/compute/cuda/kernel.h>
#include <vcl/compute/cuda/module.h>
#include <vcl/compute/cuda/platform.h>
#include <vcl/core/memory/smart_ptr.h>

// Google test
#include <gtest/gtest.h>

#ifdef VCL_CUDA_SUPPORT

class CudaFluid : public ::testing::Test
{
public:
	CudaFluid()
	{
	}

	void SetUp()
	{
		using namespace Vcl::Compute::Cuda;

		Platform::initialise();
		Platform* platform = Platform::instance();
		const Device& device = platform->device(0);
		ctx = Vcl::make_owner<Context>(device);
		ctx->bind();

		queue = Vcl::static_pointer_cast<CommandQueue>(ctx->createCommandQueue());
	}

	void TearDown()
	{
		using namespace Vcl::Compute::Cuda;

		Platform::dispose();
	}

	~CudaFluid()
	{
	}

	Vcl::owner_ptr<Vcl::Compute::Cuda::Context> ctx;
	Vcl::ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue;
};

extern uint32_t CenterGridCudaModule[];
extern size_t   CenterGridCudaModuleSize;

TEST_F(CudaFluid, LinearGridIndex)
{
	using namespace Vcl;
	using namespace Vcl::Compute::Cuda;

	// Load the module
	auto module = static_pointer_cast<Compute::Cuda::Module>(ctx->createModuleFromSource((int8_t*)CenterGridCudaModule, CenterGridCudaModuleSize * sizeof(uint32_t)));
	auto kernel = static_pointer_cast<Compute::Cuda::Kernel>(module->kernel("LinearGridIndex"));
	
	// Size of a cube size
	int s = 24;

	auto buffer = static_pointer_cast<Compute::Cuda::Buffer>(ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, s*s*s*sizeof(int4)));

	kernel->run(*queue, dim3(3, 3, 3), dim3(8, 8, 8), 0, buffer, dim3(s, s, s));

	std::vector<int4> indices(s*s*s);
	queue->read(indices.data(), { buffer }, true);

	int i = 0;
	for (int z = 0; z < s; z++)
	for (int y = 0; y < s; y++)
	for (int x = 0; x < s; x++)
	{
		int idx = i++;
		EXPECT_EQ(x, indices[idx].x);
		EXPECT_EQ(y, indices[idx].y);
		EXPECT_EQ(z, indices[idx].z);
	}
}

extern uint32_t BlockCenterGridCudaModule[];
extern size_t   BlockCenterGridCudaModuleSize;

TEST_F(CudaFluid, BlockLinearGridIndex)
{
	using namespace Vcl;
	using namespace Vcl::Compute::Cuda;

	// Load the module
	auto module = static_pointer_cast<Compute::Cuda::Module>(ctx->createModuleFromSource((int8_t*)BlockCenterGridCudaModule, BlockCenterGridCudaModuleSize * sizeof(uint32_t)));
	auto kernel = static_pointer_cast<Compute::Cuda::Kernel>(module->kernel("BlockLinearGridIndex"));
	
	// Size of a cube size
	int s = 24;

	auto buffer = static_pointer_cast<Compute::Cuda::Buffer>(ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, s*s*s*sizeof(int4)));

	kernel->run(*queue, dim3(3, 3, 3), dim3(8, 8, 8), 0, buffer, dim3(s, s, s));

	std::vector<int4> indices(s*s*s);
	queue->read(indices.data(), { buffer }, true);

	int i = 0;
	for (int z = 0; z < s; z++)
	for (int y = 0; y < s; y++)
	for (int x = 0; x < s; x++)
	{
		int idx = i++;
		EXPECT_EQ(x, indices[idx].x);
		EXPECT_EQ(y, indices[idx].y);
		EXPECT_EQ(z, indices[idx].z);
	}
}
#endif // VCL_CUDA_SUPPORT

