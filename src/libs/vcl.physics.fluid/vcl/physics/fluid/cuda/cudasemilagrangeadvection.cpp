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
#include <vcl/physics/fluid/cuda/cudasemilagrangeadvection.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/physics/fluid/cuda/cudacentergrid.h>

#ifdef VCL_CUDA_SUPPORT
extern uint32_t SemilagrangeAdvectionCudaModule[];
extern size_t SemilagrangeAdvectionCudaModuleSize;

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	Advection::Advection(ref_ptr<Compute::Cuda::Context> ctx)
	: _ownerCtx(ctx)
	{

	}
	Advection::Advection(ref_ptr<Compute::Cuda::Context> ctx, unsigned int x, unsigned int y, unsigned int z)
	: _ownerCtx(ctx)
	, _x(x)
	, _y(y)
	, _z(z)
	{

	}

	SemiLagrangeAdvection::SemiLagrangeAdvection(ref_ptr<Compute::Cuda::Context> ctx)
	: Advection(ctx)
#ifdef VCL_FLUID_ADVECT_TEX3D
	, mDevArrOldField(0)
#endif // VCL_FLUID_ADVECT_TEX3D
	{
		using namespace Vcl::Compute::Cuda;

		_advectModule = static_pointer_cast<Module>(ctx->createModuleFromSource((int8_t*) SemilagrangeAdvectionCudaModule, SemilagrangeAdvectionCudaModuleSize * sizeof(uint32_t)));

		if (_advectModule)
		{
			_advect = static_pointer_cast<Kernel>(_advectModule->kernel("AdvectSemiLagrange"));
			_advectTex = static_pointer_cast<Kernel>(_advectModule->kernel("AdvectSemiLagrangeTex"));

#ifdef VCL_FLUID_ADVECT_TEX3D
			mTexOldField = mod.textureReference("TexOldField");
			mTexOldField->setAddressMode(CU_TR_ADDRESS_MODE_CLAMP);
			mTexOldField->setFilterMode(CU_TR_FILTER_MODE_LINEAR);
			mTexOldField->setAccessMode(false, false);
#endif // VCL_FLUID_ADVECT_TEX3D
		}
	}

	SemiLagrangeAdvection::SemiLagrangeAdvection
	(
		ref_ptr<Compute::Cuda::Context> ctx,
		unsigned int x, unsigned int y, unsigned int z
	)
	: SemiLagrangeAdvection(ctx)
	{
		SemiLagrangeAdvection::setSize(x, y, z);
	}

	SemiLagrangeAdvection::~SemiLagrangeAdvection()
	{
#ifdef VCL_FLUID_ADVECT_TEX3D
		VCL_CU_SAFE_DELETE_ARRAY(mDevArrOldField);
#endif // VCL_FLUID_ADVECT_TEX3D
	}

	void SemiLagrangeAdvection::setSize(unsigned int x, unsigned int y, unsigned int z)
	{
		if (this->x() != x || this->y() != y || this->z() != z)
		{

			Advection::setSize(x, y, z);

#ifdef VCL_FLUID_ADVECT_TEX3D

			VCL_CU_SAFE_DELETE_ARRAY(mDevArrOldField);

			// Allocate a buffer for advection input
			CUDA_ARRAY3D_DESCRIPTOR desc;
			desc.Width = x;
			desc.Height = y;
			desc.Depth = z;
			desc.NumChannels = 1;
			desc.Format = CU_AD_FORMAT_FLOAT;
			desc.Flags = 0;

			VCL_CU_SAFE_CALL(cuArray3DCreate(&mDevArrOldField, &desc));
#endif // VCL_FLUID_ADVECT_TEX3D
		}
	}

	void SemiLagrangeAdvection::operator()
	(
		ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue,
		float dt,
		const Fluid::CenterGrid* g,
		ref_ptr<const Compute::Buffer> s,
		ref_ptr<Compute::Buffer> d
	)
	{
		VclRequire(dynamic_cast<const Cuda::CenterGrid*>(g), "Grid is a CUDA grid.");
		VclRequire(dynamic_pointer_cast<const Compute::Cuda::Buffer>(s), "Source is a CUDA buffer.");
		VclRequire(dynamic_pointer_cast<Compute::Cuda::Buffer>(d), "Destinatino is a CUDA buffer.");

		auto grid = static_cast<const Cuda::CenterGrid*>(g);
		auto src = static_pointer_cast<const Compute::Cuda::Buffer>(s);
		auto dst = static_pointer_cast<Compute::Cuda::Buffer>(d);
		if (!grid || !src || !dst)
			return;

#ifdef VCL_FLUID_ADVECT_TEX3D
		CUDA_MEMCPY3D cpy_desc;
		cpy_desc.WidthInBytes = x()*sizeof(float);
		cpy_desc.Height = y();
		cpy_desc.Depth = z();

		cpy_desc.srcXInBytes = 0;
		cpy_desc.srcY = 0;
		cpy_desc.srcZ = 0;
		cpy_desc.srcLOD = 0;
		cpy_desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		cpy_desc.srcDevice = src->devicePtr();
		cpy_desc.srcPitch = x()*sizeof(float);
		cpy_desc.srcHeight = y();

		cpy_desc.dstXInBytes = 0;
		cpy_desc.dstY = 0;
		cpy_desc.dstZ = 0;
		cpy_desc.dstLOD = 0;
		cpy_desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		cpy_desc.dstArray = mDevArrOldField;
		VCL_CU_SAFE_CALL(cuMemcpy3D(&cpy_desc));

		mTexOldField->bindArray(mDevArrOldField);

		// Kernel configuration
		const dim3 block_size(16, 4, 4);
		dim3 grid_size(x() / 16, y() / 4, z() / 4);

		_advectTex->configure(grid_size, block_size, 0, 0);
		_advectTex->run
		(
			dt,
			grid->velocities(0, 0).devicePtr(),
			grid->velocities(0, 1).devicePtr(),
			grid->velocities(0, 2).devicePtr(),
			dst->devicePtr(),
			dim3(x(), y(), z())
		);
#else
		// Kernel configuration
		const dim3 block_size(16, 4, 4);
		dim3 grid_size(x() / 16, y() / 4, z() / 4);

		_advect->run
		(
			*queue,
			grid_size,
			block_size,
			0,
			dt,
			grid->velocities(0, 0),
			grid->velocities(0, 1),
			grid->velocities(0, 2),
			src,
			dst,
			dim3(x(), y(), z())
		);
#endif // VCL_FLUID_ADVECT_TEX3D
	}
}}}}
#endif /* VCL_CUDA_SUPPORT */
