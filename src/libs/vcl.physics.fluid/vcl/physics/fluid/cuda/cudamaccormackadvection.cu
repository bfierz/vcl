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

#include <vcl/core/cuda/math.inc>

#include "cudasemilagrangeadvection.cu"

__global__ void AdvectMacCormack
(
	const float dt,
	const float* __restrict__ velx,
	const float* __restrict__ vely,
	const float* __restrict__ velz,
	const float* __restrict__ old_field,
	float* __restrict__ new_field,
	const dim3 res
)
{
}

extern "C"
__global__ void AdvectMacCormackMerge
(
	const float* __restrict__ phiN,
	const float* __restrict__ phiHatN,
	const float* __restrict__ phiHatN1,
	float* __restrict__ phiN1,
	const dim3 res
)
{
	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);
	const int index = idx();

	// phiN1 = phiHatN1 + (phiN - phiHatN) / 2
	phiN1[index] = phiHatN1[index] + 0.5f * (phiN[index] - phiHatN[index]);
}

extern "C"
__global__ void AdvectMacCormackClampExtrema
(
	const float dt,
	const float* __restrict__ velx,
	const float* __restrict__ vely,
	const float* __restrict__ velz,
	const float* __restrict__ old_field,
	float* __restrict__ new_field,
	const dim3 res
)
{
	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);

	const int x = idx.x();
	const int y = idx.y();
	const int z = idx.z();

	const int index = idx();

	if (0 < x && x < res.x - 1 &&
		0 < y && y < res.y - 1 &&
		0 < z && z < res.z - 1)
	{
		// Backtrace
		float xTrace = x - dt * velx[index];
		float yTrace = y - dt * vely[index];
		float zTrace = z - dt * velz[index];

		// Clamp backtrace to grid boundaries
		if (xTrace < 0.5f) xTrace = 0.5f;
		if (xTrace > res.x - 1.5f) xTrace = res.x - 1.5f;
		if (yTrace < 0.5f) yTrace = 0.5f;
		if (yTrace > res.y - 1.5f) yTrace = res.y - 1.5f;
		if (zTrace < 0.5f) zTrace = 0.5f;
		if (zTrace > res.z - 1.5f) zTrace = res.z - 1.5f;

		// Locate neighbors to interpolate
		const int x0 = (int) xTrace;
		const int x1 = x0 + 1;
		const int y0 = (int) yTrace;
		const int y1 = y0 + 1;
		const int z0 = (int) zTrace;
		const int z1 = z0 + 1;

		const int i000 = x0 + y0 * res.x + z0 * res.x * res.y;
		const int i010 = x0 + y1 * res.x + z0 * res.x * res.y;
		const int i100 = x1 + y0 * res.x + z0 * res.x * res.y;
		const int i110 = x1 + y1 * res.x + z0 * res.x * res.y;
		const int i001 = x0 + y0 * res.x + z1 * res.x * res.y;
		const int i011 = x0 + y1 * res.x + z1 * res.x * res.y;
		const int i101 = x1 + y0 * res.x + z1 * res.x * res.y;
		const int i111 = x1 + y1 * res.x + z1 * res.x * res.y;

		float minField = old_field[i000];
		float maxField = old_field[i000];

		minField = (old_field[i010] < minField) ? old_field[i010] : minField;
		maxField = (old_field[i010] > maxField) ? old_field[i010] : maxField;

		minField = (old_field[i100] < minField) ? old_field[i100] : minField;
		maxField = (old_field[i100] > maxField) ? old_field[i100] : maxField;

		minField = (old_field[i110] < minField) ? old_field[i110] : minField;
		maxField = (old_field[i110] > maxField) ? old_field[i110] : maxField;

		minField = (old_field[i001] < minField) ? old_field[i001] : minField;
		maxField = (old_field[i001] > maxField) ? old_field[i001] : maxField;

		minField = (old_field[i011] < minField) ? old_field[i011] : minField;
		maxField = (old_field[i011] > maxField) ? old_field[i011] : maxField;

		minField = (old_field[i101] < minField) ? old_field[i101] : minField;
		maxField = (old_field[i101] > maxField) ? old_field[i101] : maxField;

		minField = (old_field[i111] < minField) ? old_field[i111] : minField;
		maxField = (old_field[i111] > maxField) ? old_field[i111] : maxField;

		new_field[index] = (new_field[index] > maxField) ? maxField : new_field[index];
		new_field[index] = (new_field[index] < minField) ? minField : new_field[index];
	}
}
