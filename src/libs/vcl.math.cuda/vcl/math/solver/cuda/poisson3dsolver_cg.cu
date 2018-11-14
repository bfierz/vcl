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

__device__ void updateStencil
(
	unsigned int i, unsigned int dim, float s, float& c, float& r, float& l
)
{
	if (i < (dim - 1))
	{
		c -= s;
		r = s;
	}
	if (i > 0)
	{
		c -= s;
		l = s;
	}
}

extern "C"
__global__ void MakePoissonStencil
(
	dim3 dim,
	float h,
	float a,
	float offset,
	float* __restrict__ Ac,
	float* __restrict__ Ax_l,
	float* __restrict__ Ax_r,
	float* __restrict__ Ay_l,
	float* __restrict__ Ay_r,
	float* __restrict__ Az_l,
	float* __restrict__ Az_r,
	const unsigned char* __restrict__ skip
)
{
	const unsigned int X = dim.x;
	const unsigned int Y = dim.y;
	const unsigned int Z = dim.z;
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;
	const unsigned int index = X * Y * z + X * y + x;

	if (x < X && y < Y && z < Z)
	{
		// Scaling of the stencil
		const float s = a / (h*h);

		// Initialize write-back data
		float a_c   = 0;
		float a_x_l = 0;
		float a_x_r = 0;
		float a_y_l = 0;
		float a_y_r = 0;
		float a_z_l = 0;
		float a_z_r = 0;

		if (skip[index] == 0)
		{
			updateStencil(x, X, s, a_c, a_x_r, a_x_l);
			updateStencil(y, Y, s, a_c, a_y_r, a_y_l);
			updateStencil(z, Z, s, a_c, a_z_r, a_z_l);
		}

		Ac  [index] = offset + a_c;
		Ax_l[index] = a_x_l;
		Ax_r[index] = a_x_r;
		Ay_l[index] = a_y_l;
		Ay_r[index] = a_y_r;
		Az_l[index] = a_z_l;
		Az_r[index] = a_z_r;
	}
}

extern "C"
__global__ void ComputeInitialResidual
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
	const float* __restrict__ unknowns,

	float* __restrict__ residual,
	float* __restrict__ direction
)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;
	const unsigned int index = X * Y * z + X * y + x;

	if (x < X &&
		y < Y &&
		z < Z)
	{
		const float c = Ac[index];
		float Ax = 0.0f;
			Ax += unknowns[index] * c;
		if (x > 0)
			Ax += unknowns[index - 1] * Ax_l[index];
		if (x < X - 1)
			Ax += unknowns[index + 1] * Ax_r[index];
		if (y > 0)
			Ax += unknowns[index - X] * Ay_l[index];
		if (y < Y - 1)
			Ax += unknowns[index + X] * Ay_r[index];
		if (z > 0)
			Ax += unknowns[index - X * Y] * Az_l[index];
		if (z < Z - 1)
			Ax += unknowns[index + X * Y] * Az_r[index];

		const float r = (c != 0.0f) ? rhs[index] - Ax : 0.0f;
		residual[index]  = r;
		direction[index] = r;
	}
}

extern "C"
__global__ void ComputeQ
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
	const float* __restrict__ direction,

	float* __restrict__ q
)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;
	const unsigned int index = X * Y * z + X * y + x;

	if (x < X &&
		y < Y &&
		z < Z)
	{
		const float c = Ac[index];
		float Aq = 0.0f;
			Aq += direction[index] * c;
		if (x > 0)
			Aq += direction[index - 1] * Ax_l[index];
		if (x < X - 1)
			Aq += direction[index + 1] * Ax_r[index];
		if (y > 0)
			Aq += direction[index - X] * Ay_l[index];
		if (y < Y - 1)
			Aq += direction[index + X] * Ay_r[index];
		if (z > 0)
			Aq += direction[index - X * Y] * Az_l[index];
		if (z < Z - 1)
			Aq += direction[index + X * Y] * Az_r[index];

		Aq = (c != 0.0f) ? Aq : 0.0f;
		q[index] = Aq;
	}
}
