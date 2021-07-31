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
#include <vcl/math/solver/cuda/poisson3dsolver_cg.h>

// VCL
#include <vcl/math/solver/poisson.h>
#include <vcl/math/ceil.h>

CUresult MakePoissonStencil(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, dim3 dim, float h, float a, float offset, float* __restrict Ac, float* __restrict Ax_l, float* __restrict Ax_r, float* __restrict Ay_l, float* __restrict Ay_r, float* __restrict Az_l, float* __restrict Az_r, const unsigned char* __restrict skip);
CUresult ComputeInitialResidual(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, const unsigned int X, const unsigned int Y, const unsigned int Z, const float* __restrict Ac, const float* __restrict Ax_l, const float* __restrict Ax_r, const float* __restrict Ay_l, const float* __restrict Ay_r, const float* __restrict Az_l, const float* __restrict Az_r, const float* __restrict rhs, const float* __restrict unknowns, float* __restrict residual, float* __restrict direction);
CUresult ComputeQ(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, const unsigned int X, const unsigned int Y, const unsigned int Z, const float* __restrict Ac, const float* __restrict Ax_l, const float* __restrict Ax_r, const float* __restrict Ay_l, const float* __restrict Ay_r, const float* __restrict Az_l, const float* __restrict Az_r, const float* __restrict direction, float* __restrict q);

namespace Vcl { namespace Mathematics { namespace Solver { namespace Cuda
{
	Poisson3DCgCtx::Poisson3DCgCtx
	(
		ref_ptr<Compute::Context> ctx,
		ref_ptr<Compute::CommandQueue> queue,
		const Eigen::Vector3ui& dim
	)
	: ConjugateGradientsContext(ctx, queue, dim.x()*dim.y()*dim.z())
	, _dim(dim)
	, _unknowns(nullptr, map_t{nullptr, 0})
	, _rhs(nullptr, map_t{nullptr, 0})
	{
		using namespace Vcl::Mathematics;

		// Create buffers
		size_t size = dim.x() * dim.y() * dim.z();

		for (auto& buf : _laplacian)
			buf = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, size * sizeof(float)));
	}

	Poisson3DCgCtx::~Poisson3DCgCtx()
	{
		if (std::get<1>(_unknowns).data() != nullptr)
			_ownerCtx->release(std::get<0>(_unknowns));
		if (std::get<1>(_rhs).data() != nullptr)
			_ownerCtx->release(std::get<0>(_rhs));
	}

	void Poisson3DCgCtx::setData(map_t unknowns, const_map_t rhs)
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

		_devX = std::get<0>(_unknowns);
	}

	void Poisson3DCgCtx::setData(ref_ptr<Compute::Cuda::Buffer> unknowns, ref_ptr<Compute::Cuda::Buffer> rhs)
	{
		if (std::get<1>(_unknowns).data() != nullptr)
			_ownerCtx->release(std::get<0>(_unknowns));
		if (std::get<1>(_rhs).data() != nullptr)
			_ownerCtx->release(std::get<0>(_rhs));

		std::get<0>(_unknowns) = unknowns;
		new (&std::get<1>(_unknowns)) map_t(nullptr, 0);
		std::get<0>(_rhs) = rhs;
		new (&std::get<1>(_rhs)) const_map_t(nullptr, 0);

		_devX = std::get<0>(_unknowns);
	}

	void Poisson3DCgCtx::updatePoissonStencil(float h, float k, float o, Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip)
	{
		using Vcl::Mathematics::Solver::makePoissonStencil;

		Eigen::VectorXf Ac  { _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Ax_l{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Ax_r{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Ay_l{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Ay_r{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Az_l{ _dim.x() * _dim.y() * _dim.z() };
		Eigen::VectorXf Az_r{ _dim.x() * _dim.y() * _dim.z() };

		makePoissonStencil
		(
			_dim, h, k, o, map_t{ Ac.data(), Ac.size() },
			map_t{ Ax_l.data(), Ax_l.size() }, map_t{ Ax_r.data(), Ax_r.size() },
			map_t{ Ay_l.data(), Ay_l.size() }, map_t{ Ay_r.data(), Ay_r.size() },
			map_t{ Az_l.data(), Az_l.size() }, map_t{ Az_r.data(), Az_r.size() },
			skip
		);

		_queue->write(_laplacian[0], Ac.data(),   true);
		_queue->write(_laplacian[1], Ax_l.data(), true);
		_queue->write(_laplacian[2], Ax_r.data(), true);
		_queue->write(_laplacian[3], Ay_l.data(), true);
		_queue->write(_laplacian[4], Ay_r.data(), true);
		_queue->write(_laplacian[5], Az_l.data(), true);
		_queue->write(_laplacian[6], Az_r.data(), true);
	}

	void Poisson3DCgCtx::updatePoissonStencil(float h, float k, float o, const Compute::Cuda::Buffer& skip)
	{
		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		const dim3 block_size = { 8, 8, 4 };
		const dim3 grid_size =
		{
			ceil(_dim.x(), block_size.x) / block_size.x,
			ceil(_dim.y(), block_size.y) / block_size.y,
			ceil(_dim.z(), block_size.z) / block_size.z
		};

		VCL_CU_SAFE_CALL(MakePoissonStencil
		(
			grid_size,
			block_size,
			0,
			*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),

			// Kernel parameters
			dim3(_dim.x(), _dim.y(), _dim.z()),
			h,
			k,
			o,
			(float*) _laplacian[0]->devicePtr(),
			(float*) _laplacian[1]->devicePtr(),
			(float*) _laplacian[2]->devicePtr(),
			(float*) _laplacian[3]->devicePtr(),
			(float*) _laplacian[4]->devicePtr(),
			(float*) _laplacian[5]->devicePtr(),
			(float*) _laplacian[6]->devicePtr(),
			(const unsigned char*) skip.devicePtr()
		));
	}

	void Poisson3DCgCtx::computeInitialResidual()
	{
		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		const dim3 block_size = { 8, 8, 4 };
		const dim3 grid_size =
		{
			ceil(_dim.x(), block_size.x) / block_size.x,
			ceil(_dim.y(), block_size.y) / block_size.y,
			ceil(_dim.z(), block_size.z) / block_size.z
		};

		VCL_CU_SAFE_CALL(ComputeInitialResidual
		(
			grid_size,
			block_size,
			0,
			*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),

			// Kernel parameters
			_dim.x(),
			_dim.y(),
			_dim.z(),
			(float*) _laplacian[0]->devicePtr(),
			(float*) _laplacian[1]->devicePtr(),
			(float*) _laplacian[2]->devicePtr(),
			(float*) _laplacian[3]->devicePtr(),
			(float*) _laplacian[4]->devicePtr(),
			(float*) _laplacian[5]->devicePtr(),
			(float*) _laplacian[6]->devicePtr(),
			(float*) std::get<0>(_rhs)->devicePtr(),
			(float*) std::get<0>(_unknowns)->devicePtr(),
			(float*) _devResidual->devicePtr(),
			(float*) _devDirection->devicePtr()
		));

//#define VCL_PHYSICS_FLUID_CUDA_CG_INIT_VERIFY
#ifdef VCL_PHYSICS_FLUID_CUDA_CG_INIT_VERIFY
		
		// Compare against CPU implementation
		const unsigned int X = _dim.x();
		const unsigned int Y = _dim.y();
		const unsigned int Z = _dim.z();
		const unsigned int size = X*Y*Z;

		std::vector<float> Ac(size, 0.0f);
		std::vector<float> Ax_l(size, 0.0f);
		std::vector<float> Ax_r(size, 0.0f);
		std::vector<float> Ay_l(size, 0.0f);
		std::vector<float> Ay_r(size, 0.0f);
		std::vector<float> Az_l(size, 0.0f);
		std::vector<float> Az_r(size, 0.0f);

		std::vector<float> f(size, 0.0f);
		std::vector<float> b(size, 0.0f);
		std::vector<float> r(size, 0.0f);
		std::vector<float> d(size, 0.0f);
		
		cuMemcpyDtoH(Ac.data(),   _laplacian[0]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Ax_l.data(), _laplacian[1]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Ax_r.data(), _laplacian[2]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Ay_l.data(), _laplacian[3]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Ay_r.data(), _laplacian[4]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Az_l.data(), _laplacian[5]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Az_r.data(), _laplacian[6]->devicePtr(), size * sizeof(float));

		cuMemcpyDtoH(f.data(), std::get<0>(_unknowns)->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(b.data(), std::get<0>(_rhs)->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(r.data(), _devResidual->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(d.data(), _devDirection->devicePtr(), size * sizeof(float));

		float r_error_L1 = 0.0f;
		float d_error_L1 = 0.0f;
		size_t index = X * Y + X + 1;
		for (size_t z = 1; z < Z - 1; z++, index += 2 * X)
		{
			for (size_t y = 1; y < Y - 1; y++, index += 2)
			{
				for (size_t x = 1; x < X - 1; x++, index++)
				{
					float Ax =
						f[index + 0] * Ac[index] +
						f[index - 1] * Ax_l[index] +
						f[index + 1] * Ax_r[index] +
						f[index - X] * Ay_l[index] +
						f[index + X] * Ay_r[index] +
						f[index - X * Y] * Az_l[index] +
						f[index + X * Y] * Az_r[index];
					Ax = (Ac[index] != 0) ? Ax : 0.0f;

					// Compute error
					r[index] -= b[index] - Ax;
					d[index] -= b[index] - Ax;
					r_error_L1 += abs(r[index]);
					d_error_L1 += abs(d[index]);
				}
			}
		}

		std::cout << "L1 r: " << r_error_L1 << std::endl;
		std::cout << "L1 d: " << d_error_L1 << std::endl;
#endif // VCL_PHYSICS_FLUID_CUDA_CG_INIT_VERIFY
	}

	void Poisson3DCgCtx::computeQ()
	{
		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		const dim3 block_size = { 8, 8, 4 };
		const dim3 grid_size =
		{
			ceil(_dim.x(), block_size.x) / block_size.x,
			ceil(_dim.y(), block_size.y) / block_size.y,
			ceil(_dim.z(), block_size.z) / block_size.z
		};

		VCL_CU_SAFE_CALL(ComputeQ
		(
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
			(float*)_devDirection->devicePtr(),
			(float*)_devQ->devicePtr()
		));

//#define VCL_PHYSICS_FLUID_CUDA_CG_Q_VERIFY
#ifdef VCL_PHYSICS_FLUID_CUDA_CG_Q_VERIFY
		// Compare against CPU implementation
		const unsigned int X = _dim.x();
		const unsigned int Y = _dim.y();
		const unsigned int Z = _dim.z();
		const unsigned int size = X*Y*Z;

		std::vector<float> Ac  (size, 0.0f);
		std::vector<float> Ax_l(size, 0.0f);
		std::vector<float> Ax_r(size, 0.0f);
		std::vector<float> Ay_l(size, 0.0f);
		std::vector<float> Ay_r(size, 0.0f);
		std::vector<float> Az_l(size, 0.0f);
		std::vector<float> Az_r(size, 0.0f);

		std::vector<float> q(size, 0.0f);
		std::vector<float> d(size, 0.0f);
		std::vector<float> r(size, 0.0f);

		cuMemcpyDtoH(Ac.data(),   _laplacian[0]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Ax_l.data(), _laplacian[1]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Ax_r.data(), _laplacian[2]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Ay_l.data(), _laplacian[3]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Ay_r.data(), _laplacian[4]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Az_l.data(), _laplacian[5]->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(Az_r.data(), _laplacian[6]->devicePtr(), size * sizeof(float));

		cuMemcpyDtoH(q.data(), _devQ->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(d.data(), _devDirection->devicePtr(), size * sizeof(float));
		cuMemcpyDtoH(r.data(), _devResidual->devicePtr(), size * sizeof(float));

		float error_L1 = 0.0f;
		size_t index = X * Y + X + 1;
		for (size_t z = 1; z < Z - 1; z++, index += 2 * X)
		{
			for (size_t y = 1; y < Y - 1; y++, index += 2)
			{
				for (size_t x = 1; x < X - 1; x++, index++)
				{
					float Ad =
						d[index + 0]     * Ac[index] +
						d[index - 1]     * Ax_l[index] +
						d[index + 1]     * Ax_r[index] +
						d[index - X]     * Ay_l[index] +
						d[index + X]     * Ay_r[index] +
						d[index - X * Y] * Az_l[index] +
						d[index + X * Y] * Az_r[index];
					Ad = (Ac[index] != 0) ? Ad : 0.0f;

					// Compute error
					q[index] -= Ad;
					error_L1 += abs(q[index]);
				}
			}
		}

		std::cout << "L1 q: " << error_L1 << std::endl;

#endif // VCL_PHYSICS_FLUID_CUDA_CG_Q_VERIFY
	}
}}}}
