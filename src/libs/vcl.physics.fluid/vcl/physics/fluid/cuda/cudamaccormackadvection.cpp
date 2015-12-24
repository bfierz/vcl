/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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
#include <vcl/physics/fluid/cuda/cudamaccormackadvection.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/physics/fluid/cuda/cudacentergrid.h>

#ifdef VCL_CUDA_SUPPORT
extern uint32_t MacCormackAdvectionCudaModule [];
extern size_t MacCormackAdvectionCudaModuleSize;


namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	MacCormackAdvection::MacCormackAdvection(ref_ptr<Compute::Cuda::Context> ctx)
	: MacCormackAdvection(ctx, 16, 16, 16)
	{
	}

	MacCormackAdvection::MacCormackAdvection(ref_ptr<Compute::Cuda::Context> ctx, unsigned int x, unsigned int y, unsigned int z)
	: Advection(ctx)
	{
		using namespace Vcl::Compute::Cuda;
		using namespace std;

		_advectModule = static_pointer_cast<Module>(ctx->createModuleFromSource((int8_t*) MacCormackAdvectionCudaModule, MacCormackAdvectionCudaModuleSize * sizeof(uint32_t)));

		if (_advectModule)
		{
			_advect = static_pointer_cast<Kernel>(_advectModule->kernel("AdvectSemiLagrange"));
			_merge = static_pointer_cast<Kernel>(_advectModule->kernel("AdvectMacCormackMerge"));
			_clamp = static_pointer_cast<Kernel>(_advectModule->kernel("AdvectMacCormackClampExtrema"));
		}

		MacCormackAdvection::setSize(x, y, z);
	}

	MacCormackAdvection::~MacCormackAdvection()
	{
		context()->release(_intermediateBuffer);
	}

	void MacCormackAdvection::setSize(unsigned int x, unsigned int y, unsigned int z)
	{
		using namespace Vcl::Compute::Cuda;

		Advection::setSize(x, y, z);

		if (_intermediateBuffer)
			context()->release(_intermediateBuffer);

		unsigned int cells = x * y * z;
		unsigned int mem_size = cells * sizeof(float);
		_intermediateBuffer = static_pointer_cast<Buffer>(context()->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size));
	}

	void MacCormackAdvection::operator()
	(
		ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue,
		float dt,
		const Fluid::CenterGrid* g,
		ref_ptr<const Compute::Buffer> s,
		ref_ptr<Compute::Buffer> d
	)
	{
		Require(dynamic_cast<const Cuda::CenterGrid*>(g), "Grid is a CUDA grid.");
		Require(dynamic_pointer_cast<const Compute::Cuda::Buffer>(s), "Source is a CUDA buffer.");
		Require(dynamic_pointer_cast<Compute::Cuda::Buffer>(d), "Destinatino is a CUDA buffer.");

		auto grid = static_cast<const Cuda::CenterGrid*>(g);
		auto src = static_pointer_cast<const Compute::Cuda::Buffer>(s);
		auto dst = static_pointer_cast<Compute::Cuda::Buffer>(d);
		if (!grid || !src || !dst)
			return;

		// Fetch velocity buffers 
		auto vel_x = grid->velocities(0, 0);
		auto vel_y = grid->velocities(0, 1);
		auto vel_z = grid->velocities(0, 2);

		auto phiN  = src;
		auto phiN1 = dst;

		auto phiHatN = dst;
		auto phiHatN1 = _intermediateBuffer;

		// phiHatN1 = A(phiN)
		const dim3 block(16, 4, 4);
		dim3 gridConfig(x() / 16, y() / 4, z() / 4);

		_advect->run
		(
			*queue,
			gridConfig,
			block,
			0,
			dt,
			vel_x, vel_y, vel_z,
			phiN,
			phiHatN1,
			dim3(x(), y(), z())
		);

		// phiHatN = A^R(phiHatN1)
		_advect->run
		(
			*queue,
			gridConfig,
			block,
			0,
			-1.0f * dt,
			vel_x, vel_y, vel_z,
			phiHatN1,
			phiHatN,
			dim3(x(), y(), z())
		);

		// phiN1 = phiHatN1 + (phiN - phiHatN) / 2
		_merge->run
		(
			*queue,
			gridConfig,
			block,
			0,
			phiN,
			phiHatN,
			phiHatN1,
			phiN1,
			dim3(x(), y(), z())
		);

		grid->copyBorderX(*queue, *dst, Eigen::Vector3i(x(), y(), z()));
		grid->copyBorderY(*queue, *dst, Eigen::Vector3i(x(), y(), z()));
		grid->copyBorderZ(*queue, *dst, Eigen::Vector3i(x(), y(), z()));
		
		// Clamp any newly created extrema
		_clamp->run
		(
			*queue,
			gridConfig,
			block,
			0,
			dt,
			vel_x, vel_y, vel_z,
			phiN, phiN1,
			dim3(x(), y(), z())
		);
		
		// If the error estimate was bad, revert to first order
		//clampOutsideRays(dt, vel_x, vel_y, vel_z, phiN, phiN1, phiHatN1);
	}
}}}}
#endif // VCL_CUDA_SUPPORT
