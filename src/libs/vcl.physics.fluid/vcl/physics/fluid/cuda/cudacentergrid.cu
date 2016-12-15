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
#if !(defined(__cplusplus) && defined(__CUDACC__))
#	error "Compile with CUDA compiler"
#endif

#include <vcl/core/cuda/common.inc>
#include <vcl/physics/fluid/cuda/centergrid.inc>

extern "C"
__global__ void AccumulateField
(
	float* __restrict__ dst,
	const float* __restrict__ src,
	float alpha,
	int n
)
{
	// Indices
	int i0;

	// Compute the index for the first element to fetch
	i0 = blockIdx.x*blockDim.x + threadIdx.x;

	if (i0 < n)
	{
		dst[i0] += alpha * src[i0];
	}
}

/*!
 *	\brief Copy the borders in x-direction to ensure a Neumann boundary condition
 *
 *	The mode parameter configures how the border is set:
 *	- 0 - Set a diriclet boundary condition
 *	- 1 - Set the border to the adjacent element
 *	- 2 - Set a Neumann boundary condition
 */
extern "C"
__global__ void SetBorderX
(
	int mode,
	float* Field,
	const dim3 dim
)
{
	int y = blockIdx.x*blockDim.x + threadIdx.x;
	int z = blockIdx.y*blockDim.y + threadIdx.y;

	if (y < dim.y && z < dim.z)
	{
		int index = y * dim.x + z * dim.x * dim.y;

		if (mode == 0)
		{
			// left slab
			Field[index] = 0.0f;

			// right slab
			index += dim.x - 1;
			Field[index] = 0.0f;
		}
		else if (mode == 1)
		{
			// left slab
			Field[index] = Field[index + 1];

			// right slab
			index += dim.x - 1;
			Field[index] = Field[index - 1];
		}
		else if(mode == 2)
		{
			// left slab
			Field[index] = Field[index + 2];

			// right slab
			index += dim.x - 1;
			Field[index] = Field[index - 2];
		}
	}
}

extern "C"
__global__ void SetBorderY
(
	int mode,
	float* Field,
	const dim3 dim
)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int z = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < dim.x && z < dim.z)
	{
		int index = x + z * dim.x * dim.y;

		if (mode == 0)
		{
			// left slab
			Field[index] = 0.0f;

			// right slab
			index += dim.x * dim.y - dim.x;
			Field[index] = 0.0f;
		}
		else if (mode == 1)
		{
			// left slab
			Field[index] = Field[index + 1 * dim.x];

			// right slab
			index += dim.x * dim.y - dim.x;
			Field[index] = Field[index - 1 * dim.x];
		}
		else if (mode == 2)
		{
			// left slab
			Field[index] = Field[index + 2 * dim.x];

			// right slab
			index += dim.x * dim.y - dim.x;
			Field[index] = Field[index - 2 * dim.x];
		}
	}
}

extern "C"
__global__ void SetBorderZ
(
	int mode,
	float* Field,
	const dim3 dim
)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < dim.x && y < dim.y)
	{
		int index = x + y * dim.x;

		if (mode == 0)
		{
			// left slab
			Field[index] = 0.0f;

			// right slab
			index += dim.x * dim.y * dim.z - dim.x * dim.y;
			Field[index] = 0.0f;
		}
		else if (mode == 1)
		{
			// left slab
			Field[index] = Field[index + 1 * dim.x * dim.y];

			// right slab
			index += dim.x * dim.y * dim.z - dim.x * dim.y;
			Field[index] = Field[index - 1 * dim.x * dim.y];
		}
		else if (mode == 2)
		{
			// left slab
			Field[index] = Field[index + 2 * dim.x * dim.y];

			// right slab
			index += dim.x * dim.y * dim.z - dim.x * dim.y;
			Field[index] = Field[index - 2 * dim.x * dim.y];
		}
	}
}
