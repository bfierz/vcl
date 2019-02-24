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
#include <vcl/core/cuda/common.inc>

template<typename T, int I, int J>
class MatrixArrayStorage
{
public:
	__device__
	MatrixArrayStorage(T* data, int n, int idx)
	{
		_idx = idx;
		_n = n;
		_data = data;
	}

public:
	__device__
	T& operator() (int i)
	{
		return _data[_n*i + _idx];
	}

	__device__
	T& operator() (int i, int j)
	{
		return _data[_n*(j*I + i) + _idx];
	}

private:
	int _idx;
	int _n;
	T* _data;
};

// McAdams SVD library
#define USE_SCALAR_IMPLEMENTATION
#define HAS_RSQRT

#define COMPUTE_V_AS_MATRIX
#define COMPUTE_U_AS_MATRIX

#include <vcl/math/mcadams/Singular_Value_Decomposition_Preamble.hpp>

extern "C"
__global__ void JacobiSVD33McAdams
(
	int size,
	int capacity,
	const float* __restrict__ memA,
	float* __restrict__ memU,
	float* __restrict__ memV,
	float* __restrict__ memS
)
{
#define JACOBI_CONJUGATION_SWEEPS (int) 4

	int globalIdx = threadIdx.x + blockDim.x*blockIdx.x;
	if (globalIdx >= size)
		return;

	MatrixArrayStorage<const float, 3, 3> A(memA, capacity, globalIdx);
	MatrixArrayStorage<float, 3, 3> U(memU, capacity, globalIdx);
	MatrixArrayStorage<float, 3, 3> V(memV, capacity, globalIdx);
	MatrixArrayStorage<float, 3, 1> S(memS, capacity, globalIdx);

#include <vcl/math/mcadams/Singular_Value_Decomposition_Kernel_Declarations.hpp>

	Sa11.f = A(0, 0);
	Sa21.f = A(1, 0);
	Sa31.f = A(2, 0);
	Sa12.f = A(0, 1);
	Sa22.f = A(1, 1);
	Sa32.f = A(2, 1);
	Sa13.f = A(0, 2);
	Sa23.f = A(1, 2);
	Sa33.f = A(2, 2);

#include <vcl/math/mcadams/Singular_Value_Decomposition_Main_Kernel_Body.hpp>

	U(0, 0) = Su11.f;
	U(1, 0) = Su21.f;
	U(2, 0) = Su31.f;
	U(0, 1) = Su12.f;
	U(1, 1) = Su22.f;
	U(2, 1) = Su32.f;
	U(0, 2) = Su13.f;
	U(1, 2) = Su23.f;
	U(2, 2) = Su33.f;

	V(0, 0) = Sv11.f;
	V(1, 0) = Sv21.f;
	V(2, 0) = Sv31.f;
	V(0, 1) = Sv12.f;
	V(1, 1) = Sv22.f;
	V(2, 1) = Sv32.f;
	V(0, 2) = Sv13.f;
	V(1, 2) = Sv23.f;
	V(2, 2) = Sv33.f;

	S(0) = Sa11.f;
	S(1) = Sa22.f;
	S(2) = Sa33.f;
}
