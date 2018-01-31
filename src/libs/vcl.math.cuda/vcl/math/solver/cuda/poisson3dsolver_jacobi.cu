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
#include <vcl/core/cuda/common.inc>
#include <vcl/core/cuda/math.inc>

extern "C"
__global__ void PoissonUpdateSolution
(
	const unsigned int X,
	const unsigned int Y,
	const unsigned int Z,

	const float* __restrict__ Ac,
	const float* __restrict__ Ax_l,
	const float* __restrict__ Ax_r,
	const float* __restrict__ Ay_l,
	const float* __restrict__ Ay_r,
	const float* __restrict__ Az_l,
	const float* __restrict__ Az_r,
	const float* __restrict__ rhs,

	float* __restrict__ unknowns,
	float* __restrict__ next,
	float* __restrict__ error
)
{
	// x^{n+1} = D^-1 (b - R x^{n})
	//                -------------
	//                      q

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;
	const unsigned int index = X*Y * z + X * y + x;

	if (0 < x && x < X-1 &&
		0 < y && y < Y-1 &&
		0 < z && z < Z-1   )
	{
		float q =
			unknowns[index - 1] * Ax_l[index] +
			unknowns[index + 1] * Ax_r[index] +
			unknowns[index - X] * Ay_l[index] +
			unknowns[index + X] * Ay_r[index] +
			unknowns[index - X * Y] * Az_l[index] +
			unknowns[index + X * Y] * Az_r[index];

		const float c = Ac[index];
		float n = (rhs[index] - q) / c;
		n = (c != 0) ? n : unknowns[index];

		next[index] = n;

		// Compute the error
		if (c)
		{
			const float e = rhs[index] - (Ac[index] * unknowns[index] + q);
			atomicAdd(error, e * e);
		}
	}
}
