/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <vcl/math/solver/cuda/poisson3dsolver_jacobi.h>

// VCL
#include <vcl/math/ceil.h>

CUresult MakePoissonStencil(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, dim3 dim, float h, float a, float offset, float* __restrict Ac, float* __restrict Ax_l, float* __restrict Ax_r, float* __restrict Ay_l, float* __restrict Ay_r, float* __restrict Az_l, float* __restrict Az_r, const unsigned char* __restrict skip);
CUresult PoissonUpdateSolution(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, const unsigned int X, const unsigned int Y, const unsigned int Z, const float* __restrict Ac, const float* __restrict Ax_l, const float* __restrict Ax_r, const float* __restrict Ay_l, const float* __restrict Ay_r, const float* __restrict Az_l, const float* __restrict Az_r, const float* __restrict rhs, float* __restrict unknowns, float* __restrict next, float* __restrict error);

namespace Vcl { namespace Mathematics { namespace Solver { namespace Cuda {
	Poisson3DJacobiCtx::Poisson3DJacobiCtx(
		ref_ptr<Compute::Context> ctx,
		ref_ptr<Compute::CommandQueue> queue,
		const Eigen::Vector3ui& dim)
	: _ownerCtx(ctx)
	, _queue(queue)
	, _dim(dim)
	, _unknowns(nullptr, map_t{ nullptr, 0 })
	, _rhs(nullptr, map_t{ nullptr, 0 })
	{
		using namespace Vcl::Mathematics;

		// Create buffers
		size_t size = dim.x() * dim.y() * dim.z();

		for (auto& buf : _laplacian)
			buf = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, size * sizeof(float)));

		_next = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, size * sizeof(float)));
		_dev_error = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, sizeof(float)));
	}

	Poisson3DJacobiCtx::~Poisson3DJacobiCtx()
	{
		for (auto& buf : _laplacian)
			_ownerCtx->release(buf);

		_ownerCtx->release(_next);

		if (std::get<1>(_unknowns).data() != nullptr)
			_ownerCtx->release(std::get<0>(_unknowns));
		if (std::get<1>(_rhs).data() != nullptr)
			_ownerCtx->release(std::get<0>(_rhs));
	}

	void Poisson3DJacobiCtx::setData(map_t unknowns, const_map_t rhs)
	{
		if (std::get<1>(_unknowns).data() != nullptr)
			_ownerCtx->release(std::get<0>(_unknowns));
		if (std::get<1>(_rhs).data() != nullptr)
			_ownerCtx->release(std::get<0>(_rhs));

		std::get<0>(_unknowns) = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, size() * sizeof(float)));
		new (&std::get<1>(_unknowns)) map_t(unknowns);
		std::get<0>(_rhs) = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, size() * sizeof(float)));
		new (&std::get<1>(_rhs)) const_map_t(rhs);

		_queue->write(std::get<0>(_unknowns), unknowns.data(), true);
		_queue->write(std::get<0>(_rhs), rhs.data(), true);
	}

	void Poisson3DJacobiCtx::setData(ref_ptr<Compute::Cuda::Buffer> unknowns, ref_ptr<Compute::Cuda::Buffer> rhs)
	{
		if (std::get<1>(_unknowns).data() != nullptr)
			_ownerCtx->release(std::get<0>(_unknowns));
		if (std::get<1>(_rhs).data() != nullptr)
			_ownerCtx->release(std::get<0>(_rhs));

		std::get<0>(_unknowns) = unknowns;
		new (&std::get<1>(_unknowns)) map_t(nullptr, 0);
		std::get<0>(_rhs) = rhs;
		new (&std::get<1>(_rhs)) const_map_t(nullptr, 0);
	}

	void Poisson3DJacobiCtx::updatePoissonStencil(float h, float k, float o, Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip)
	{
		Eigen::VectorXf Ac{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Ax_l{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Ax_r{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Ay_l{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Ay_r{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Az_l{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Az_r{ _dim.x() * _dim.y() * _dim.z() };

		makePoissonStencil(
			_dim, h, k, o, map_t{ Ac.data(), Ac.size() },
			map_t{ Ax_l.data(), Ax_l.size() }, map_t{ Ax_r.data(), Ax_r.size() },
			map_t{ Ay_l.data(), Ay_l.size() }, map_t{ Ay_r.data(), Ay_r.size() },
			map_t{ Az_l.data(), Az_l.size() }, map_t{ Az_r.data(), Az_r.size() },
			skip);

		_queue->write(_laplacian[0], Ac.data(), true);
		_queue->write(_laplacian[1], Ax_l.data(), true);
		_queue->write(_laplacian[2], Ax_r.data(), true);
		_queue->write(_laplacian[3], Ay_l.data(), true);
		_queue->write(_laplacian[4], Ay_r.data(), true);
		_queue->write(_laplacian[5], Az_l.data(), true);
		_queue->write(_laplacian[6], Az_r.data(), true);
	}

	void Poisson3DJacobiCtx::updatePoissonStencil(float h, float k, float o, const Compute::Cuda::Buffer& skip)
	{
		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		const dim3 block_size = { 8, 8, 4 };
		const dim3 grid_size = {
			ceil(_dim.x(), block_size.x) / block_size.x,
			ceil(_dim.y(), block_size.y) / block_size.y,
			ceil(_dim.z(), block_size.z) / block_size.z
		};

		VCL_CU_SAFE_CALL(MakePoissonStencil(
			grid_size,
			block_size,
			0,
			*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),

			// Kernel parameters
			dim3(_dim.x(), _dim.y(), _dim.z()),
			h,
			k,
			o,
			(float*)_laplacian[0]->devicePtr(),
			(float*)_laplacian[1]->devicePtr(),
			(float*)_laplacian[2]->devicePtr(),
			(float*)_laplacian[3]->devicePtr(),
			(float*)_laplacian[4]->devicePtr(),
			(float*)_laplacian[5]->devicePtr(),
			(float*)_laplacian[6]->devicePtr(),
			(const unsigned char*)skip.devicePtr()));
	}

	int Poisson3DJacobiCtx::size() const
	{
		return _dim.x() * _dim.y() * _dim.z();
	}

	void Poisson3DJacobiCtx::precompute()
	{
		_error = 0;
	}

	void Poisson3DJacobiCtx::updateSolution()
	{
		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		const dim3 block_size = { 8, 8, 4 };
		const dim3 grid_size = {
			ceil(_dim.x(), block_size.x) / block_size.x,
			ceil(_dim.y(), block_size.y) / block_size.y,
			ceil(_dim.z(), block_size.z) / block_size.z
		};

		// Reset error counter
		float zero = 0.0f;
		_queue->fill(_dev_error, &zero, sizeof(float));

		VCL_CU_SAFE_CALL(PoissonUpdateSolution(
			grid_size,
			block_size,
			0,
			*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),

			// Kernel parameters
			_dim.x(),
			_dim.y(),
			_dim.z(),
			(float*)_laplacian[0]->devicePtr(),
			(float*)_laplacian[1]->devicePtr(),
			(float*)_laplacian[2]->devicePtr(),
			(float*)_laplacian[3]->devicePtr(),
			(float*)_laplacian[4]->devicePtr(),
			(float*)_laplacian[5]->devicePtr(),
			(float*)_laplacian[6]->devicePtr(),
			(float*)std::get<0>(_rhs)->devicePtr(),

			(float*)std::get<0>(_unknowns)->devicePtr(),
			(float*)_next->devicePtr(),
			(float*)_dev_error->devicePtr()));

		_queue->copy(std::get<0>(_unknowns), _next);
	}

	double Poisson3DJacobiCtx::computeError()
	{
		_queue->read(&_error, _dev_error, true);
		return sqrt(_error) / size();
	}

	void Poisson3DJacobiCtx::finish(double* residual)
	{
		if (std::get<1>(_unknowns).data() != nullptr)
			_queue->read(std::get<1>(_unknowns).data(), std::get<0>(_unknowns), true);

		if (residual)
		{
			_queue->read(&_error, _dev_error, true);
			*residual = _error;
		}
	}
}}}}
