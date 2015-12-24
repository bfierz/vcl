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
#include <vcl/config/eigen.h>

// C++ standard library

// Boost

// VCL library
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/device.h>
#include <vcl/compute/cuda/platform.h>

#include <vcl/physics/fluid/cuda/cudacentergrid.h>
#include <vcl/physics/fluid/cuda/cudaeulerfluidsimulation.h>
#include <vcl/physics/fluid/cuda/cudaenergydecomposition.h>
#include <vcl/physics/fluid/cuda/cudawaveletturbulencefluidsimulation.h>

#include "bitmap.h"

int main(int argc, char* argv[])
{
	VCL_UNREFERENCED_PARAMETER(argc);
	VCL_UNREFERENCED_PARAMETER(argv);

#ifdef VCL_CUDA_SUPPORT
	Vcl::Compute::Cuda::Platform::initialise();
	Vcl::Compute::Cuda::Platform* platform = Vcl::Compute::Cuda::Platform::instance();
	const Vcl::Compute::Cuda::Device& device = platform->device(0);
	auto ctx = Vcl::make_owner<Vcl::Compute::Cuda::Context>(device);
	ctx->bind();

	int c = 128;
	Vcl::Physics::Fluid::Cuda::CenterGrid grid(ctx, Vcl::static_pointer_cast<Vcl::Compute::Cuda::CommandQueue>(ctx->defaultQueue()), Eigen::Vector3i(c, c, c), 1.0f);
	grid.setBuoyancy(1);
	grid.setVorticityCoeff(0.2f);
	Vcl::Physics::Fluid::Cuda::EulerFluidSimulation solver(ctx);
	Vcl::Physics::Fluid::Cuda::EnergyDecomposition energy(ctx);
	//Vcl::Physics::Fluid::Cuda::WaveletTrubulenceFluidSimulation noise(context);

	std::vector<std::array<unsigned char, 4>> bitmap;
	bitmap.resize(c * c);
	for (int i = 0; i < 10; i++)
	{
		solver.update(grid, 0.016);
		energy.compute(grid);
		
		std::vector<float> density(grid.densities(0)->size() / sizeof(float), 0.0f);
		ctx->defaultQueue()->read(density.data(), grid.densities(0), true);

		for (int z = 0; z < c; z++)
		{
			for (int x = 0; x < c; x++)
			{
				unsigned char d = (unsigned char) (density[z * c * c + c/2 * c + x]*12.5);
				bitmap[z * c + x][0] = d;
				bitmap[z * c + x][1] = d;
				bitmap[z * c + x][2] = d;
				bitmap[z * c + x][3] = d;
			}
		}
		
		Vcl::IO::Bitmap::store("density" + std::to_string(i) + ".bmp", c, c, bitmap);
	}
#endif // VCL_CUDA_SUPPORT

	return 0;
}
