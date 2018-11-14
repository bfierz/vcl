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
#if VCL_MATH_CG_CUDA_SHUFFLE_ATOMICS
		_reduceKernel    = static_pointer_cast<Compute::Cuda::Kernel>(_reduceUpdateModule->kernel("CGComputeReductionShuffleAtomics"));
#elif VCL_MATH_CG_CUDA_SHUFFLE
		_reduceBeginKernel    = static_pointer_cast<Compute::Cuda::Kernel>(_reduceUpdateModule->kernel("CGComputeReductionShuffleBegin"));
		_reduceContinueKernel = static_pointer_cast<Compute::Cuda::Kernel>(_reduceUpdateModule->kernel("CGComputeReductionShuffleContinue"));
#elif VCL_MATH_CG_CUDA_BASIC
		_reduceBeginKernel    = static_pointer_cast<Compute::Cuda::Kernel>(_reduceUpdateModule->kernel("CGComputeReductionBegin"));
		_reduceContinueKernel = static_pointer_cast<Compute::Cuda::Kernel>(_reduceUpdateModule->kernel("CGComputeReductionContinue"));
#endif
		_updateKernel         = static_pointer_cast<Compute::Cuda::Kernel>(_reduceUpdateModule->kernel("CGUpdateVectorsEx"));
		
		init();
	}

	ConjugateGradientsContext::~ConjugateGradientsContext()
	{
		destroy();
	}

	void ConjugateGradientsContext::init()
	{
#if VCL_MATH_CG_CUDA_SHUFFLE_ATOMICS
		const unsigned int gridSize = 1;
#else
		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		// Process 4 vector entries per thread
		const unsigned int blockSize = 128;
		const unsigned int elemPerThread = 4;
		const unsigned int elemPerBlock = elemPerThread * blockSize;
		const unsigned int gridSize = ceil(_size, elemPerBlock) / elemPerBlock;
#endif

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
		VclRequire(dynamic_pointer_cast<Compute::Cuda::Buffer>(x), "x is CUDA buffer.");

		_devX = static_pointer_cast<Compute::Cuda::Buffer>(x);
	}

	void ConjugateGradientsContext::reduceVectors()
	{
		// Reduction kernels used in this method are based on
		// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		// Process 4 vector entries per thread
		const unsigned int blockSize = 128;
		const unsigned int elemPerThread = 4;
		const unsigned int elemPerBlock = elemPerThread * blockSize;
		const unsigned int gridSize = ceil(_size, elemPerBlock) / elemPerBlock;		
#if VCL_MATH_CG_CUDA_SHUFFLE
		const unsigned int dynamicSharedMemory = 0;
#elif VCL_MATH_CG_CUDA_BASIC
		const unsigned int dynamicSharedMemory = elemPerBlock * sizeof(float);
#endif
		// Initialise the reduction
#if VCL_MATH_CG_CUDA_SHUFFLE_ATOMICS
		_reduceKernel->run
		(
			*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),
			gridSize,
			blockSize,
			0,
			
			// Kernel parameters
			_size,
			_devResidual,
			_devDirection,
			_devQ,
			_reduceBuffersR[0],
			_reduceBuffersG[0],
			_reduceBuffersB[0],
			_reduceBuffersA[0]
		);
#elif VCL_MATH_CG_CUDA_SHUFFLE || VCL_MATH_CG_CUDA_BASIC
		_reduceBeginKernel->run
		(
			*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),
			gridSize,
			blockSize,
			dynamicSharedMemory,
			
			// Kernel parameters
			_size,
			_devResidual,
			_devDirection,
			_devQ,
			_reduceBuffersR[0],
			_reduceBuffersG[0],
			_reduceBuffersB[0],
			_reduceBuffersA[0]
		);
		
		// Reduce the array of partial results to a scalar.
		// The initial reduction step produced 'gridSize' number of
		// partial results.
		unsigned int nrPartialResults = gridSize;
		while (nrPartialResults > 1)
		{
			unsigned int currGridSize = ceil(nrPartialResults, elemPerBlock) / elemPerBlock;
			_reduceContinueKernel->run
			(
				*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),
				currGridSize,
				blockSize,
				dynamicSharedMemory,
				
				// Kernel parameters
				nrPartialResults,
				_reduceBuffersR[0], _reduceBuffersG[0], _reduceBuffersB[0], _reduceBuffersA[0],
				_reduceBuffersR[1], _reduceBuffersG[1], _reduceBuffersB[1], _reduceBuffersA[1]
			);

			// Update next loop.
			// Each block produced an output element.
			nrPartialResults = currGridSize;
	
			// Swap output to input
			std::swap(_reduceBuffersR[0], _reduceBuffersR[1]);
			std::swap(_reduceBuffersG[0], _reduceBuffersG[1]);
			std::swap(_reduceBuffersB[0], _reduceBuffersB[1]);
			std::swap(_reduceBuffersA[0], _reduceBuffersA[1]);
		}
#endif
	}

	void ConjugateGradientsContext::updateVectors()
	{
		VclRequire(_devX, "Solution vector is set.");

		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		const unsigned int blockSize = 256;
		const unsigned int elemPerThread = 4;
		const unsigned int elemPerBlock = elemPerThread * blockSize;
		const unsigned int gridSize = ceil(_size, elemPerBlock) / elemPerBlock;

		// Update the vectors
		static_pointer_cast<Compute::Cuda::Kernel>(_updateKernel)->run
		(
			*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),
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

		float beta = *_hostR - (2.0f * alpha) * *_hostB + (alpha * alpha) * *_hostA;
		return abs(beta);
	}

	void ConjugateGradientsContext::finish(double* residual)
	{
		if (residual)
		{
			float alpha = 0.0f;
			if (abs(*_hostG) > 0.0f)
				alpha = *_hostR / *_hostG;

			float beta = *_hostR - (2.0f * alpha) * *_hostB + (alpha * alpha) * *_hostA;
			*residual = abs(beta);
		}
	}
}}}}
#endif // VCL_CUDA_SUPPORT
