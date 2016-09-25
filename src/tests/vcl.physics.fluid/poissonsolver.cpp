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

#include <vcl/physics/fluid/cuda/cudapoisson3dsolver_jacobi.h>

// Google test
#include <gtest/gtest.h>

#ifdef VCL_CUDA_SUPPORT
extern uint32_t Poisson3dSolverCudaModule[];
extern size_t Poisson3dSolverCudaModuleSize;

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

void buildLaplacian
(
	std::vector<float>& Ac,
	std::vector<float>& Ax,
	std::vector<float>& Ay,
	std::vector<float>& Az,
	const std::vector<int>& obstacles,
	Eigen::Vector3i dim
)
{
	std::fill(Ac.begin(), Ac.end(), 0.0f);
	std::fill(Ax.begin(), Ax.end(), 0.0f);
	std::fill(Ay.begin(), Ay.end(), 0.0f);
	std::fill(Az.begin(), Az.end(), 0.0f);

	size_t index = 0;
	for (size_t z = 0; z < dim.z(); z++)
	{
		for (size_t y = 0; y < dim.y(); y++)
		{
			for (size_t x = 0; x < dim.x(); x++, index++)
			{
				if (!obstacles[index])
				{
					// Set the matrix to the Poisson stencil in order
					if (x < (dim.x() - 1) && !obstacles[index + 1])
					{
						Ac[index] += 1.0;
						Ax[index] = -1.0;
					}
					if (x > 0 && !obstacles[index - 1])
					{
						Ac[index] += 1.0;
					}
					if (y < (dim.y() - 1) && !obstacles[index + dim.x()])
					{
						Ac[index] += 1.0;
						Ay[index] = -1.0;
					}
					if (y > 0 && !obstacles[index - dim.x()])
					{
						Ac[index] += 1.0;
					}
					if (z < (dim.z() - 1) && !obstacles[index + dim.x()*dim.y()])
					{
						Ac[index] += 1.0;
						Az[index] = -1.0;
					}
					if (z > 0 && !obstacles[index - dim.x()*dim.y()])
					{
						Ac[index] += 1.0;
					}
				}
			}
		}
	}
}

TEST_F(CudaFluid, BuildLaplacianWithoutObstacles)
{
	using namespace Vcl;
	using namespace Vcl::Compute::Cuda;
	using namespace Vcl::Physics::Fluid::Cuda;

	// Load the module
	auto module = static_pointer_cast<Compute::Cuda::Module>(ctx->createModuleFromSource((int8_t*)Poisson3dSolverCudaModule, Poisson3dSolverCudaModuleSize * sizeof(uint32_t)));
	
	// Load the related kernels
	//auto compDiv = static_pointer_cast<Compute::Cuda::Kernel>(module->kernel("ComputeDivergence"));
	auto buildLhs = static_pointer_cast<Compute::Cuda::Kernel>(module->kernel("BuildLHS"));
	//auto correctField = static_pointer_cast<Compute::Cuda::Kernel>(module->kernel("CorrectVelocities"));

	int x = 32;
	int y = 32;
	int z = 32;

	dim3 dimension(x, y, z);
	dim3 block_size(16, 8, 1);
	dim3 grid_size(x / block_size.x, y / block_size.y, z / block_size.z);

	int cells = x * y * z;
	size_t mem_size = cells * sizeof(float);

	auto laplacian0 = static_pointer_cast<Compute::Cuda::Buffer>(ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); queue->setZero(laplacian0);
	auto laplacian1 = static_pointer_cast<Compute::Cuda::Buffer>(ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); queue->setZero(laplacian1);
	auto laplacian2 = static_pointer_cast<Compute::Cuda::Buffer>(ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); queue->setZero(laplacian2);
	auto laplacian3 = static_pointer_cast<Compute::Cuda::Buffer>(ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); queue->setZero(laplacian3);
	auto obstacles  = static_pointer_cast<Compute::Cuda::Buffer>(ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); queue->setZero(obstacles);

	buildLhs->run
	(
		*queue,
		grid_size,
		block_size,
		0,
		laplacian0,
		laplacian1,
		laplacian2,
		laplacian3,
		obstacles,
		0,
		false,
		dimension
	);

	std::vector<float> l0(cells);
	std::vector<float> l1(cells);
	std::vector<float> l2(cells);
	std::vector<float> l3(cells);
	queue->read(l0.data(), { laplacian0 }, true);
	queue->read(l1.data(), { laplacian1 }, true);
	queue->read(l2.data(), { laplacian2 }, true);
	queue->read(l3.data(), { laplacian3 }, true);

	std::vector<float> a0(cells);
	std::vector<float> a1(cells);
	std::vector<float> a2(cells);
	std::vector<float> a3(cells);
	std::vector<int>   o (cells, 0);

	buildLaplacian(a0, a1, a2, a3, o, { x, y, z });

	EXPECT_TRUE(std::equal(a0.begin(), a0.end(), l0.begin()));
	EXPECT_TRUE(std::equal(a1.begin(), a1.end(), l1.begin()));
	EXPECT_TRUE(std::equal(a2.begin(), a2.end(), l2.begin()));
	EXPECT_TRUE(std::equal(a3.begin(), a3.end(), l3.begin()));
}

#endif // VCL_CUDA_SUPPORT

