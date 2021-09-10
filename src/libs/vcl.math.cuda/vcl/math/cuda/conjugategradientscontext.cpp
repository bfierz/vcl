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

#ifdef VCL_CUDA_SUPPORT
CUresult CGComputeReductionBegin(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, unsigned int n, const float* __restrict g_r, const float* __restrict g_d, const float* __restrict g_q, float* __restrict d_r, float* __restrict d_g, float* __restrict d_b, float* __restrict d_a);
CUresult CGComputeReductionContinue(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, unsigned int n, const float* __restrict in_d_r, const float* __restrict in_d_g, const float* __restrict in_d_b, const float* __restrict in_d_a, float* __restrict out_d_r, float* __restrict out_d_g, float* __restrict out_d_b, float* __restrict out_d_a);
CUresult CGComputeReductionShuffleBegin(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, unsigned int n, const float* __restrict g_r, const float* __restrict g_d, const float* __restrict g_q, float* __restrict d_r, float* __restrict d_g, float* __restrict d_b, float* __restrict d_a);
CUresult CGComputeReductionShuffleContinue(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, unsigned int n, const float* __restrict in_d_r, const float* __restrict in_d_g, const float* __restrict in_d_b, const float* __restrict in_d_a, float* __restrict out_d_r, float* __restrict out_d_g, float* __restrict out_d_b, float* __restrict out_d_a);
CUresult CGComputeReductionShuffleAtomics(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, unsigned int n, const float* __restrict g_r, const float* __restrict g_d, const float* __restrict g_q, float* __restrict d_r, float* __restrict d_g, float* __restrict d_b, float* __restrict d_a);
CUresult CGUpdateVectorsEx(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream, const unsigned int n, const float* __restrict d_r_ptr, const float* __restrict d_g_ptr, const float* __restrict d_b_ptr, const float* __restrict d_a_ptr, float* __restrict vX, float* __restrict vD, float* __restrict vQ, float* __restrict vR);

namespace Vcl { namespace Mathematics { namespace Solver { namespace Cuda {
	ConjugateGradientsContext::ConjugateGradientsContext(
		ref_ptr<Compute::Context> ctx,
		ref_ptr<Compute::CommandQueue> queue,
		int size)
	: _ownerCtx(ctx)
	, _queue(queue)
	, _size(size)
	{
		init();
	}

	ConjugateGradientsContext::~ConjugateGradientsContext()
	{
		destroy();
	}

	void ConjugateGradientsContext::init()
	{
#	if VCL_MATH_CG_CUDA_SHUFFLE_ATOMICS
		const unsigned int gridSize = 1;
#	else
		// Compute block and grid size
		// Has to be multiple of 16 (memory alignment) and 32 (warp size)
		// Process 4 vector entries per thread
		const unsigned int blockSize = 128;
		const unsigned int elemPerThread = 4;
		const unsigned int elemPerBlock = elemPerThread * blockSize;
		const unsigned int gridSize = ceil(_size, elemPerBlock) / elemPerBlock;
#	endif

		// Create buffers
		_devDirection = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, _size * sizeof(float)));
		_devQ = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, _size * sizeof(float)));
		_devResidual = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, _size * sizeof(float)));

		_reduceBuffersR[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize * sizeof(float)));
		_reduceBuffersG[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize * sizeof(float)));
		_reduceBuffersB[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize * sizeof(float)));
		_reduceBuffersA[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize * sizeof(float)));

		_reduceBuffersR[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize * sizeof(float)));
		_reduceBuffersG[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize * sizeof(float)));
		_reduceBuffersB[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize * sizeof(float)));
		_reduceBuffersA[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::None, gridSize * sizeof(float)));

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
		VCL_CU_SAFE_CALL(cuMemHostAlloc((void**)&_hostR, sizeof(float), 0));
		VCL_CU_SAFE_CALL(cuMemHostAlloc((void**)&_hostG, sizeof(float), 0));
		VCL_CU_SAFE_CALL(cuMemHostAlloc((void**)&_hostB, sizeof(float), 0));
		VCL_CU_SAFE_CALL(cuMemHostAlloc((void**)&_hostA, sizeof(float), 0));
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
#	if VCL_MATH_CG_CUDA_SHUFFLE
		const unsigned int dynamicSharedMemory = 0;
#	elif VCL_MATH_CG_CUDA_BASIC
		const unsigned int dynamicSharedMemory = elemPerBlock * sizeof(float);
#	endif
		// Initialise the reduction
#	if VCL_MATH_CG_CUDA_SHUFFLE_ATOMICS
		CGComputeReductionShuffleAtomics(
			gridSize,
			blockSize,
			0,
			*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),

			// Kernel parameters
			_size,
			(float*)_devResidual->devicePtr(),
			(float*)_devDirection->devicePtr(),
			(float*)_devQ->devicePtr(),
			(float*)_reduceBuffersR[0]->devicePtr(),
			(float*)_reduceBuffersG[0]->devicePtr(),
			(float*)_reduceBuffersB[0]->devicePtr(),
			(float*)_reduceBuffersA[0]->devicePtr());
#	elif VCL_MATH_CG_CUDA_SHUFFLE || VCL_MATH_CG_CUDA_BASIC
#		ifdef VCL_MATH_CG_CUDA_SHUFFLE
		CGComputeReductionShuffleBegin
#		elif defined VCL_MATH_CG_CUDA_BASIC
		CGComputeReductionBegin
#		endif
			(
				gridSize,
				blockSize,
				dynamicSharedMemory,
				*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),

				// Kernel parameters
				_size,
				(float*)_devResidual->devicePtr(),
				(float*)_devDirection->devicePtr(),
				(float*)_devQ->devicePtr(),
				(float*)_reduceBuffersR[0]->devicePtr(),
				(float*)_reduceBuffersG[0]->devicePtr(),
				(float*)_reduceBuffersB[0]->devicePtr(),
				(float*)_reduceBuffersA[0]->devicePtr());

		// Reduce the array of partial results to a scalar.
		// The initial reduction step produced 'gridSize' number of
		// partial results.
		unsigned int nrPartialResults = gridSize;
		while (nrPartialResults > 1)
		{
			unsigned int currGridSize = ceil(nrPartialResults, elemPerBlock) / elemPerBlock;
#		ifdef VCL_MATH_CG_CUDA_SHUFFLE
			CGComputeReductionShuffleContinue
#		elif defined VCL_MATH_CG_CUDA_BASIC
			CGComputeReductionContinue
#		endif
				(
					currGridSize,
					blockSize,
					dynamicSharedMemory,
					*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),

					// Kernel parameters
					nrPartialResults,
					(float*)_reduceBuffersR[0]->devicePtr(),
					(float*)_reduceBuffersG[0]->devicePtr(),
					(float*)_reduceBuffersB[0]->devicePtr(),
					(float*)_reduceBuffersA[0]->devicePtr(),
					(float*)_reduceBuffersR[1]->devicePtr(),
					(float*)_reduceBuffersG[1]->devicePtr(),
					(float*)_reduceBuffersB[1]->devicePtr(),
					(float*)_reduceBuffersA[1]->devicePtr());

			// Update next loop.
			// Each block produced an output element.
			nrPartialResults = currGridSize;

			// Swap output to input
			std::swap(_reduceBuffersR[0], _reduceBuffersR[1]);
			std::swap(_reduceBuffersG[0], _reduceBuffersG[1]);
			std::swap(_reduceBuffersB[0], _reduceBuffersB[1]);
			std::swap(_reduceBuffersA[0], _reduceBuffersA[1]);
		}
#	endif
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
		CGUpdateVectorsEx(
			gridSize,
			blockSize,
			0,
			*static_pointer_cast<Compute::Cuda::CommandQueue>(_queue),

			_size,
			(float*)_reduceBuffersR[0]->devicePtr(),
			(float*)_reduceBuffersG[0]->devicePtr(),
			(float*)_reduceBuffersB[0]->devicePtr(),
			(float*)_reduceBuffersA[0]->devicePtr(),
			(float*)_devX->devicePtr(),
			(float*)_devDirection->devicePtr(),
			(float*)_devQ->devicePtr(),
			(float*)_devResidual->devicePtr());
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
