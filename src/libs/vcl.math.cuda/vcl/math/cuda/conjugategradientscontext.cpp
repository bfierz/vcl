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
#include <vcl/math/cuda/conjugategradientscontext.h>

// VCL
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/kernel.h>
#include <vcl/math/ceil.h>

// Kernels
extern uint32_t CGCtxCU[];
extern size_t CGCtxCUSize;

#ifdef VCL_CUDA_SUPPORT
namespace Vcl { namespace Mathematics { namespace Solver { namespace Cuda
{
	ConjugateGradientsContext::ConjugateGradientsContext
	(
		ref_ptr<Compute::Context> ctx,
		ref_ptr<Compute::CommandQueue> queue,
		int size
	)
	: _ownerCtx(ctx)
	, _queue(queue)
	, _size(size)
	{
		using namespace Vcl::Mathematics;
		
		// Load the module
		_reduceUpdateModule = ctx->createModuleFromSource(reinterpret_cast<const int8_t*>(CGCtxCU), CGCtxCUSize * sizeof(uint32_t));

		// Load kernels
		_reduceBeginKernel    = static_pointer_cast<Compute::Cuda::Kernel>(_reduceUpdateModule->kernel("CGComputeReductionBegin"));
		_reduceContinueKernel = static_pointer_cast<Compute::Cuda::Kernel>(_reduceUpdateModule->kernel("CGComputeReductionContinue"));
		_updateKernel         = static_pointer_cast<Compute::Cuda::Kernel>(_reduceUpdateModule->kernel("CGUpdateVectorsEx"));
		
		init();
	}

	ConjugateGradientsContext::~ConjugateGradientsContext()
	{
		destroy();
	}

	void ConjugateGradientsContext::init()
	{
		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		unsigned int blockSize = 128;
		unsigned int elemPerThread = 1;
		unsigned int elemPerBlock = elemPerThread * blockSize;
		unsigned int gridSize = ceil<1 * 128>(_size) / (elemPerBlock);

		// Create buffers
		_devDirection = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, _size*sizeof(float)));
		_devQ = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, _size*sizeof(float)));
		_devResidual = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, _size*sizeof(float)));

		_reduceBuffersR[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize*sizeof(float)));
		_reduceBuffersG[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize*sizeof(float)));
		_reduceBuffersB[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize*sizeof(float)));
		_reduceBuffersA[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize*sizeof(float)));

		_reduceBuffersR[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize*sizeof(float)));
		_reduceBuffersG[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize*sizeof(float)));
		_reduceBuffersB[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize*sizeof(float)));
		_reduceBuffersA[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize*sizeof(float)));

		// Initialise buffers
		_queue->setZero(_devDirection);
		_queue->setZero(_devQ);
		_queue->setZero(_devResidual);

		_queue->setZero(_reduceBuffersR[0]);
		_queue->setZero(_reduceBuffersG[0]);
		_queue->setZero(_reduceBuffersB[0]);
		_queue->setZero(_reduceBuffersA[0]);

		_queue->setZero(_reduceBuffersR[1]);
		_queue->setZero(_reduceBuffersG[1]);
		_queue->setZero(_reduceBuffersB[1]);
		_queue->setZero(_reduceBuffersA[1]);

		// Register the memory on to allow asynchronous memory transfers
		VCL_CU_SAFE_CALL(cuMemHostAlloc((void**) &_hostR, sizeof(float), 0));
		VCL_CU_SAFE_CALL(cuMemHostAlloc((void**) &_hostG, sizeof(float), 0));
		VCL_CU_SAFE_CALL(cuMemHostAlloc((void**) &_hostB, sizeof(float), 0));
		VCL_CU_SAFE_CALL(cuMemHostAlloc((void**) &_hostA, sizeof(float), 0));
	}

	void ConjugateGradientsContext::destroy()
	{
		_ownerCtx->release(_devDirection);
		_ownerCtx->release(_devQ);
		_ownerCtx->release(_devResidual);

		_ownerCtx->release(_reduceBuffersR[0]);
		_ownerCtx->release(_reduceBuffersG[0]);
		_ownerCtx->release(_reduceBuffersB[0]);
		_ownerCtx->release(_reduceBuffersA[0]);
		_ownerCtx->release(_reduceBuffersR[1]);
		_ownerCtx->release(_reduceBuffersG[1]);
		_ownerCtx->release(_reduceBuffersB[1]);
		_ownerCtx->release(_reduceBuffersA[1]);

		cuMemFreeHost(_hostR);
		cuMemFreeHost(_hostG);
		cuMemFreeHost(_hostB);
		cuMemFreeHost(_hostA);
	}

	int ConjugateGradientsContext::size() const
	{
		return _size;
	}

	void ConjugateGradientsContext::setX(ref_ptr<Compute::Buffer> x)
	{
		Require(dynamic_pointer_cast<Compute::Cuda::Buffer>(x), "x is CUDA buffer.");

		_devX = static_pointer_cast<Compute::Cuda::Buffer>(x);
	}

	void ConjugateGradientsContext::reduceVectors()
	{
		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		unsigned int blockSize = 128;
		unsigned int elemPerThread = 4;
		unsigned int elemPerBlock = elemPerThread * blockSize;
		unsigned int gridSize = ceil<4 * 128>(_size) / (elemPerBlock);

		// Initialise the reduction
		Core::static_pointer_cast<Compute::Cuda::Kernel>(_reduceBeginKernel)->run
		(
			*Core::static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),
			gridSize,
			blockSize,
			4 * blockSize * sizeof(float),
			_size,
			_devResidual,
			_devDirection,
			_devQ,
			_reduceBuffersR[0],
			_reduceBuffersG[0],
			_reduceBuffersB[0],
			_reduceBuffersA[0]
		);
		
		// Reduce the array to a scalar
		unsigned int n = _size / elemPerBlock;
		while (n > 1)
		{
			Core::static_pointer_cast<Compute::Cuda::Kernel>(_reduceBeginKernel)->run
			(
				*Core::static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),
				std::max(n / elemPerBlock, (unsigned int) 1),
				blockSize,
				4 * blockSize * sizeof(float),
				n,
				_reduceBuffersR[0], _reduceBuffersG[0], _reduceBuffersB[0], _reduceBuffersA[0],
				_reduceBuffersR[1], _reduceBuffersG[1], _reduceBuffersB[1], _reduceBuffersA[1]
			);

			// Update next loop
			n = n / elemPerBlock;
	
			// Swap output to input
			std::swap(_reduceBuffersR[0], _reduceBuffersR[1]);
			std::swap(_reduceBuffersG[0], _reduceBuffersG[1]);
			std::swap(_reduceBuffersB[0], _reduceBuffersB[1]);
			std::swap(_reduceBuffersA[0], _reduceBuffersA[1]);
		}


		// Compare against CPU implementation
		//std::vector<float> r(mSize, 0.0f);
		//std::vector<float> d(mSize, 0.0f);
		//std::vector<float> q(mSize, 0.0f);
		//
		//std::vector<float> d_r(gridSize, 0.0f);
		//std::vector<float> d_g(gridSize, 0.0f);
		//std::vector<float> d_b(gridSize, 0.0f);
		//std::vector<float> d_a(gridSize, 0.0f);
		//
		//cuMemcpyDtoH(r.data(), mDevResidual, r.size() * sizeof(float));
		//cuMemcpyDtoH(d.data(), mDevDirection, d.size() * sizeof(float));
		//cuMemcpyDtoH(q.data(), mDevQ, q.size() * sizeof(float));
		//
		//cuMemcpyDtoH(d_r.data(), mDevReduceBuffersR[0], d_r.size()*sizeof(float));
		//cuMemcpyDtoH(d_g.data(), mDevReduceBuffersG[0], d_g.size()*sizeof(float));
		//cuMemcpyDtoH(d_b.data(), mDevReduceBuffersB[0], d_b.size()*sizeof(float));
		//cuMemcpyDtoH(d_a.data(), mDevReduceBuffersA[0], d_a.size()*sizeof(float));
		//
		//float l1_r = 0;
		//float l1_g = 0;
		//float l1_b = 0;
		//float l1_a = 0;
		//for (size_t i = 0; i < mSize; i++)
		//{
		//	l1_r += r[i] * r[i];
		//	l1_g += d[i] * q[i];
		//	l1_b += r[i] * q[i];
		//	l1_a += q[i] * q[i];
		//}
		//
		//std::cout
		//	<< "L1: "
		//	<< l1_r - d_r[0] << ", "
		//	<< l1_g - d_g[0] << ", "
		//	<< l1_b - d_b[0] << ", "
		//	<< l1_a - d_a[0] << ", "
		//	<< std::endl;
	}

	void ConjugateGradientsContext::updateVectors()
	{
		Require(_devX, "Solution vector is set.");

		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		unsigned int blockSize = 256;
		unsigned int elemPerThread = 4;
		unsigned int elemPerBlock = elemPerThread * blockSize;
		unsigned int gridSize = ceil<4*256>(_size) / (elemPerBlock);

		// Update the vectors
		Core::static_pointer_cast<Compute::Cuda::Kernel>(_updateKernel)->run
		(
			*Core::static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),
			gridSize,
			blockSize,
			0,
			_size,
			_reduceBuffersR[0], _reduceBuffersG[0], _reduceBuffersB[0], _reduceBuffersA[0],
			_devX, _devDirection, _devQ, _devResidual
		);
	}

	double ConjugateGradientsContext::computeError()
	{
		_queue->read(_hostR, Compute::BufferView(_reduceBuffersR[0], 0, sizeof(float)));
		_queue->read(_hostG, Compute::BufferView(_reduceBuffersG[0], 0, sizeof(float)));
		_queue->read(_hostA, Compute::BufferView(_reduceBuffersA[0], 0, sizeof(float)));
		_queue->read(_hostB, Compute::BufferView(_reduceBuffersB[0], 0, sizeof(float)));
		_queue->sync();
				
		float alpha = 0.0f;
		if (abs(*_hostG) > 0.0f)
			alpha = *_hostR / *_hostG;

		float beta = *_hostR - 2.0f * alpha * *_hostB + alpha * alpha * *_hostA;
		if (abs(*_hostR) > 0.0f)
			beta = beta / *_hostR;

		return abs(beta * *_hostR);
	}

	void ConjugateGradientsContext::finish(double* residual)
	{
		if (residual)
			*residual = *_hostR;
	}
}}}}
#endif /* VCL_CUDA_SUPPORT */
