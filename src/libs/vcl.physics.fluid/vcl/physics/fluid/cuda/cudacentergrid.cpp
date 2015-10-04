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
#include <vcl/physics/fluid/cuda/cudacentergrid.h>

// VCL
#include <vcl/compute/cuda/context.h>
#include <vcl/compute/cuda/module.h>
#include <vcl/core/contract.h>
#include <vcl/math/ceil.h>

#ifdef VCL_CUDA_SUPPORT
extern uint32_t CenterGridCudaModule[];
extern size_t CenterGridCudaModuleSize;

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	CenterGrid::CenterGrid
	(
		ref_ptr<Vcl::Compute::Cuda::Context> ctx,
		ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue,
		const Eigen::Vector3i& resolution,
		float spacing
	)
	: Fluid::CenterGrid(resolution, spacing, -1)
	, _ownerCtx(ctx)
	{
		Require(ctx, "Context is valid.");

		size_t bufferSize = resolution.x()*resolution.z()*resolution.z()*sizeof(float);

		_force[0] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_force[0]);
		_force[1] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_force[1]);
		_force[2] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_force[2]);

		_velocity[0][0] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_velocity[0][0]);
		_velocity[0][1] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_velocity[0][1]);
		_velocity[0][2] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_velocity[0][2]);

		_velocity[1][0] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_velocity[1][0]);
		_velocity[1][1] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_velocity[1][1]);
		_velocity[1][2] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_velocity[1][2]);

		_vorticityMag = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_vorticityMag);
		_vorticity[0] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_vorticity[0]);
		_vorticity[1] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_vorticity[1]);
		_vorticity[2] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_vorticity[2]);

		_obstacles            = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_obstacles);
		_obstaclesVelocity[0] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_obstaclesVelocity[0]);
		_obstaclesVelocity[1] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_obstaclesVelocity[1]);
		_obstaclesVelocity[2] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_obstaclesVelocity[2]);

		_density[0] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_density[0]);
		_density[1] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_density[1]);

		_heat[0] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_heat[0]);
		_heat[1] = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, bufferSize); queue->setZero(_heat[1]);

		// Load the module
		_fluidModule = ctx->createModuleFromSource((int8_t*) CenterGridCudaModule, CenterGridCudaModuleSize*sizeof(uint32_t));

		if (_fluidModule)
		{
			// Load the fluid simulation related kernels
			_setBorderX = Core::static_pointer_cast<Compute::Cuda::Kernel>(_fluidModule->kernel("SetBorderX"));
			_setBorderY = Core::static_pointer_cast<Compute::Cuda::Kernel>(_fluidModule->kernel("SetBorderY"));
			_setBorderZ = Core::static_pointer_cast<Compute::Cuda::Kernel>(_fluidModule->kernel("SetBorderZ"));

			_accumulate = Core::static_pointer_cast<Compute::Cuda::Kernel>(_fluidModule->kernel("AccumulateField"));
		}
	}

	CenterGrid::~CenterGrid()
	{
		_ownerCtx->release(_force[0]);
		_ownerCtx->release(_force[1]);
		_ownerCtx->release(_force[2]);

		_ownerCtx->release(_velocity[0][0]);
		_ownerCtx->release(_velocity[0][1]);
		_ownerCtx->release(_velocity[0][2]);

		_ownerCtx->release(_velocity[1][0]);
		_ownerCtx->release(_velocity[1][1]);
		_ownerCtx->release(_velocity[1][2]);

		_ownerCtx->release(_vorticityMag);
		_ownerCtx->release(_vorticity[0]);
		_ownerCtx->release(_vorticity[1]);
		_ownerCtx->release(_vorticity[2]);

		_ownerCtx->release(_obstacles);
		_ownerCtx->release(_obstaclesVelocity[0]);
		_ownerCtx->release(_obstaclesVelocity[1]);
		_ownerCtx->release(_obstaclesVelocity[2]);

		_ownerCtx->release(_density[0]);
		_ownerCtx->release(_density[1]);

		_ownerCtx->release(_heat[0]);
		_ownerCtx->release(_heat[1]);
	}

	void CenterGrid::resize(const Eigen::Vector3i& resolution)
	{

	}

	void CenterGrid::swap()
	{
		std::swap(_velocity[0], _velocity[1]);
		std::swap(_density[0],  _density[1]);
		std::swap(_heat[0],     _heat[1]);
	}

	CenterGrid::ref_ptr<Compute::Buffer> CenterGrid::aquireIntermediateBuffer()
	{
		if (_freeTmpBuffers.size() > 0)
		{
			auto handle = _freeTmpBuffers.back();
			_freeTmpBuffers.pop_back();

			_aquiredTmpBuffers.insert(handle);
			return handle;
		}
		else
		{
			size_t bufferSize = resolution().x()*resolution().z()*resolution().z()*sizeof(float);
			auto handle = _ownerCtx->createBuffer(Compute::BufferAccess::None, bufferSize);

			_aquiredTmpBuffers.insert(handle);
			return handle;
		}
	}
	void CenterGrid::releaseIntermediateBuffer(ref_ptr<Compute::Buffer> handle)
	{
		auto entry = _aquiredTmpBuffers.find(handle);
		if (entry != _aquiredTmpBuffers.end())
		{
			auto handle = *entry;

			_aquiredTmpBuffers.erase(entry);
			_freeTmpBuffers.push_back(handle);
		}
	}

	bool CenterGrid::hasNamedBuffer(const std::string& name) const
	{
		return _namedBuffers.find(name) != _namedBuffers.end();
	}
	CenterGrid::ref_ptr<Compute::Buffer> CenterGrid::namedBuffer(const std::string& name) const
	{
		Require(hasNamedBuffer(name), "Buffer exists.");

		return _namedBuffers.find(name)->second;
	}
	CenterGrid::ref_ptr<Compute::Buffer> CenterGrid::addNamedBuffer(const std::string& name)
	{
		Require(!hasNamedBuffer(name), "Buffer does not exist.");

		size_t bufferSize = resolution().x()*resolution().z()*resolution().z()*sizeof(float);
		auto handle = _ownerCtx->createBuffer(Compute::BufferAccess::None, bufferSize);

		_namedBuffers[name] = handle;
		return handle;
	}

	Compute::Cuda::Buffer& CenterGrid::forces(int dim)
	{
		Require(0 <= dim && dim < 3, "Dimension is valid.");

		return *Core::static_pointer_cast<Compute::Cuda::Buffer>(_force[dim]);
	}

	Compute::Cuda::Buffer& CenterGrid::velocities(int i, int dim)
	{
		Require(0 <= i && i < 2, "Index is valid.");
		Require(0 <= dim && dim < 3, "Dimension is valid.");

		return *Core::static_pointer_cast<Compute::Cuda::Buffer>(_velocity[i][dim]);
	}

	const Compute::Cuda::Buffer& CenterGrid::velocities(int i, int dim) const
	{
		Require(0 <= i && i < 2, "Index is valid.");
		Require(0 <= dim && dim < 3, "Dimension is valid.");

		return *Core::static_pointer_cast<Compute::Cuda::Buffer>(_velocity[i][dim]);
	}

	CenterGrid::ref_ptr<Compute::Buffer> CenterGrid::obstacles() const
	{
		return _obstacles;
	}
	Compute::Cuda::Buffer&  CenterGrid::obstacleBuffer()
	{
		return *Core::static_pointer_cast<Compute::Cuda::Buffer>(_obstacles);
	}

	CenterGrid::ref_ptr<Compute::Buffer> CenterGrid::densities(int i) const
	{
		Require(0 <= i && i < 2, "Index is valid.");

		return _density[i];
	}
	Compute::Cuda::Buffer&  CenterGrid::densityBuffer(int i)
	{
		Require(0 <= i && i < 2, "Index is valid.");

		return *Core::static_pointer_cast<Compute::Cuda::Buffer>(_density[i]);
	}

	CenterGrid::ref_ptr<Compute::Buffer> CenterGrid::heat(int i) const
	{
		Require(0 <= i && i < 2, "Index is valid.");

		return _heat[i];
	}
	Compute::Cuda::Buffer&  CenterGrid::heatBuffer(int i)
	{
		Require(0 <= i && i < 2, "Index is valid.");

		return *Core::static_pointer_cast<Compute::Cuda::Buffer>(_heat[i]);
	}

	Compute::Cuda::Buffer& CenterGrid::vorticityMag()
	{
		return *Core::static_pointer_cast<Compute::Cuda::Buffer>(_vorticityMag);
	}
	Compute::Cuda::Buffer& CenterGrid::vorticity(int dim)
	{
		Require(0 <= dim && dim < 3, "Index is valid.");

		return *Core::static_pointer_cast<Compute::Cuda::Buffer>(_vorticity[dim]);
	}

	void CenterGrid::setBorderZero(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderX(queue, BorderOp::SetZero, buffer, dim);
		setBorderY(queue, BorderOp::SetZero, buffer, dim);
		setBorderZ(queue, BorderOp::SetZero, buffer, dim);
	}

	void CenterGrid::setBorderZeroX(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderX(queue, BorderOp::SetZero, buffer, dim);
	}
	void CenterGrid::setBorderZeroY(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderY(queue, BorderOp::SetZero, buffer, dim);
	}
	void CenterGrid::setBorderZeroZ(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderZ(queue, BorderOp::SetZero, buffer, dim);
	}

	void CenterGrid::copyBorderX(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderX(queue, BorderOp::Copy, buffer, dim);
	}
	void CenterGrid::copyBorderY(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderY(queue, BorderOp::Copy, buffer, dim);
	}
	void CenterGrid::copyBorderZ(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderZ(queue, BorderOp::Copy, buffer, dim);
	}

	void CenterGrid::setNeumannX(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderX(queue, BorderOp::Neumann, buffer, dim);
	}
	void CenterGrid::setNeumannY(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderY(queue, BorderOp::Neumann, buffer, dim);
	}
	void CenterGrid::setNeumannZ(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		setBorderZ(queue, BorderOp::Neumann, buffer, dim);
	}

	void CenterGrid::setBorderX(Vcl::Compute::Cuda::CommandQueue& queue, BorderOp mode, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		dim3 block_size(16, 16);
		dim3 grid_size(dim.y() / block_size.x, dim.z() / block_size.y);

		_setBorderX->run
		(
			queue,
			grid_size,
			block_size,
			0,
			(int) mode,
			(CUdeviceptr) buffer,
			dim
		);
	}

	void CenterGrid::setBorderY(Vcl::Compute::Cuda::CommandQueue& queue, BorderOp mode, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		dim3 block_size(16, 16);
		dim3 grid_size(dim.x() / block_size.x, dim.z() / block_size.y);

		_setBorderY->run
		(
			queue,
			grid_size,
			block_size,
			0,
			(int) mode,
			(CUdeviceptr) buffer,
			dim
		);
	}

	void CenterGrid::setBorderZ(Vcl::Compute::Cuda::CommandQueue& queue, BorderOp mode, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const
	{
		dim3 block_size(16, 16);
		dim3 grid_size(dim.x() / block_size.x, dim.y() / block_size.y);

		_setBorderZ->run
		(
			queue,
			grid_size,
			block_size,
			0,
			(int) mode,
			(CUdeviceptr) buffer,
			dim
		);
	}

	void CenterGrid::accumulate(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& dst, const Compute::Cuda::Buffer& src, float alpha) const
	{
		using namespace Vcl::Mathematics;

		int size = resolution().x() * resolution().y() * resolution().z();

		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		unsigned int block_size = 256;
		unsigned int elemPerThread = 1;
		unsigned int elemPerBlock = elemPerThread * block_size;
		unsigned int grid_size = ceil<1 * 256>(size) / (elemPerBlock);

		_accumulate->run
		(
			queue,
			grid_size,
			block_size,
			0,
			(CUdeviceptr) dst,
			(CUdeviceptr) src,
			alpha,
			size
		);
	}
}}}}
#endif /* VCL_CUDA_SUPPORT */
