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
#include <vcl/core/cuda/math.inc>

template<typename T, int I, int J, int NRBANKS>
class DynamicStridedCache
{
public:
	__device__
	DynamicStridedCache(int offset, int bank_idx, int sm_row)
	{
		mData = SharedMemory<T>() + offset + sm_row*I*NRBANKS + bank_idx;
	}

public:
	__device__
	T& operator() (int i)
	{
		return mData[NRBANKS*i];
	}

	__device__
	T& operator() (int i, int j)
	{
		return (mData + i)[NRBANKS*j];
	}

private:
	T* mData;
};

//#define UNROLL_LAST_32

__device__ inline float4 initialReduction
(
	unsigned int n,
	const float* __restrict__ g_r,
	const float* __restrict__ g_d,
	const float* __restrict__ g_q
)
{
	// Collect up to 4 elements and perform partial dot-products
    const unsigned int i0 = 4*blockIdx.x*blockDim.x + threadIdx.x;
    const float r0 = (i0 < n) ? g_r[i0] : 0;
    const float d0 = (i0 < n) ? g_d[i0] : 0;
    const float q0 = (i0 < n) ? g_q[i0] : 0;

	const unsigned int i1 = i0 + blockDim.x;
	const float r1 = (i1 < n) ? g_r[i1] : 0;
	const float d1 = (i1 < n) ? g_d[i1] : 0;
	const float q1 = (i1 < n) ? g_q[i1] : 0;
	
	const unsigned int i2 = i1 + blockDim.x;
	const float r2 = (i2 < n) ? g_r[i2] : 0;
	const float d2 = (i2 < n) ? g_d[i2] : 0;
	const float q2 = (i2 < n) ? g_q[i2] : 0;
	
	const unsigned int i3 = i2 + blockDim.x;
	const float r3 = (i3 < n) ? g_r[i3] : 0;
	const float d3 = (i3 < n) ? g_d[i3] : 0;
	const float q3 = (i3 < n) ? g_q[i3] : 0;
    
	const float rr = r0*r0 + r1*r1 + r2*r2 + r3*r3;
	const float dq = d0*q0 + d1*q1 + d2*q2 + d3*q3;
	const float rq = r0*q0 + r1*q1 + r2*q2 + r3*q3;
	const float qq = q0*q0 + q1*q1 + q2*q2 + q3*q3;
 
	return make_float4(rr, dq, rq, qq);
}

__device__ inline float4 reduction
(	
	unsigned int n,
	const float* __restrict__ d_r,
	const float* __restrict__ d_g,
	const float* __restrict__ d_b,
	const float* __restrict__ d_a
)
{
	// Collect up to 4 elements
    const unsigned int i0 = 4*blockIdx.x*blockDim.x + threadIdx.x;
    const float r0 = (i0 < n) ? d_r[i0] : 0;
    const float g0 = (i0 < n) ? d_g[i0] : 0;
    const float b0 = (i0 < n) ? d_b[i0] : 0;
    const float a0 = (i0 < n) ? d_a[i0] : 0;
	
	const unsigned int i1 = i0 + blockDim.x;
	const float r1 = (i1 < n) ? d_r[i1] : 0;
	const float g1 = (i1 < n) ? d_g[i1] : 0;
	const float b1 = (i1 < n) ? d_b[i1] : 0;
	const float a1 = (i1 < n) ? d_a[i1] : 0;
	
	const unsigned int i2 = i1 + blockDim.x;
	const float r2 = (i2 < n) ? d_r[i2] : 0;
	const float g2 = (i2 < n) ? d_g[i2] : 0;
	const float b2 = (i2 < n) ? d_b[i2] : 0;
	const float a2 = (i2 < n) ? d_a[i2] : 0;
	
	const unsigned int i3 = i2 + blockDim.x;
	const float r3 = (i3 < n) ? d_r[i3] : 0;
	const float g3 = (i3 < n) ? d_g[i3] : 0;
	const float b3 = (i3 < n) ? d_b[i3] : 0;
	const float a3 = (i3 < n) ? d_a[i3] : 0;

	const float rr = r0 + r1 + r2 + r3;
	const float dq = g0 + g1 + g2 + g3;
	const float rq = b0 + b1 + b2 + b3;
	const float qq = a0 + a1 + a2 + a3;
    
	return make_float4(rr, dq, rq, qq);
}

extern "C"
__global__ void CGComputeReductionBegin
(
	unsigned int n,
	const float* __restrict__ g_r,
	const float* __restrict__ g_d,
	const float* __restrict__ g_q,
	float* __restrict__ d_r,
	float* __restrict__ d_g,
	float* __restrict__ d_b,
	float* __restrict__ d_a
)
{
    // Load partial results to shared memory
    unsigned int tid = threadIdx.x;
	float* sd_r = SharedMemory<float>() + 0*blockDim.x;
	float* sd_g = SharedMemory<float>() + 1*blockDim.x;
	float* sd_b = SharedMemory<float>() + 2*blockDim.x;
	float* sd_a = SharedMemory<float>() + 3*blockDim.x;

	const float4 sums = initialReduction(n, g_r, g_d, g_q);
	sd_r[tid] = sums.x;
	sd_g[tid] = sums.y;
	sd_b[tid] = sums.z;
	sd_a[tid] = sums.w;
    
    __syncthreads();
    
	// Do the reductions in shared memory
#ifdef UNROLL_LAST_32
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) 
	{
		if (tid < s) 
		{
			sd_r[tid] += sd_r[tid + s];
			sd_g[tid] += sd_g[tid + s];
			sd_b[tid] += sd_b[tid + s];
			sd_a[tid] += sd_a[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		sd_r[tid] += sd_r[tid + 32];
		sd_g[tid] += sd_g[tid + 32];
		sd_b[tid] += sd_b[tid + 32];
		sd_a[tid] += sd_a[tid + 32];

		sd_r[tid] += sd_r[tid + 16];
		sd_g[tid] += sd_g[tid + 16];
		sd_b[tid] += sd_b[tid + 16];
		sd_a[tid] += sd_a[tid + 16];

		sd_r[tid] += sd_r[tid +  8];
		sd_g[tid] += sd_g[tid +  8];
		sd_b[tid] += sd_b[tid +  8];
		sd_a[tid] += sd_a[tid +  8];

		sd_r[tid] += sd_r[tid +  4];
		sd_g[tid] += sd_g[tid +  4];
		sd_b[tid] += sd_b[tid +  4];
		sd_a[tid] += sd_a[tid +  4];

		sd_r[tid] += sd_r[tid +  2];
		sd_g[tid] += sd_g[tid +  2];
		sd_b[tid] += sd_b[tid +  2];
		sd_a[tid] += sd_a[tid +  2];

		sd_r[tid] += sd_r[tid +  1];
		sd_g[tid] += sd_g[tid +  1];
		sd_b[tid] += sd_b[tid +  1];
		sd_a[tid] += sd_a[tid + 1];
	}

#else
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            sd_r[tid] += sd_r[tid + s];
            sd_g[tid] += sd_g[tid + s];
            sd_b[tid] += sd_b[tid + s];
            sd_a[tid] += sd_a[tid + s];
        }
        __syncthreads();
    }
#endif // UNROLL_LAST_32

    // Write result for this block to global memory
    if (tid == 0)
    {
		d_r[blockIdx.x] = sd_r[0];
		d_g[blockIdx.x] = sd_g[0];
		d_b[blockIdx.x] = sd_b[0];
		d_a[blockIdx.x] = sd_a[0];
	}
}


extern "C"
__global__ void CGComputeReductionContinue
(	
	unsigned int n,
	const float* __restrict__ in_d_r,
	const float* __restrict__ in_d_g,
	const float* __restrict__ in_d_b,
	const float* __restrict__ in_d_a,
	float* __restrict__ out_d_r,
	float* __restrict__ out_d_g,
	float* __restrict__ out_d_b,
	float* __restrict__ out_d_a
)
{
    // Load partial results to shared memory
    unsigned int tid = threadIdx.x;
	float* sd_r = SharedMemory<float>() + 0*blockDim.x;
	float* sd_g = SharedMemory<float>() + 1*blockDim.x;
	float* sd_b = SharedMemory<float>() + 2*blockDim.x;
	float* sd_a = SharedMemory<float>() + 3*blockDim.x;
	
	const float4 sums = reduction(n, in_d_r, in_d_g, in_d_b, in_d_a);
	sd_r[tid] = sums.x;
	sd_g[tid] = sums.y;
	sd_b[tid] = sums.z;
	sd_a[tid] = sums.w;
    
    __syncthreads();
    
	// Do the reductions in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            sd_r[tid] += sd_r[tid + s];
            sd_g[tid] += sd_g[tid + s];
            sd_b[tid] += sd_b[tid + s];
            sd_a[tid] += sd_a[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
    {
		out_d_r[blockIdx.x] = sd_r[0];
		out_d_g[blockIdx.x] = sd_g[0];
		out_d_b[blockIdx.x] = sd_b[0];
		out_d_a[blockIdx.x] = sd_a[0];
	}
}


extern "C"
__global__ void CGComputeReductionShuffleBegin
(
	unsigned int n,
	const float* __restrict__ g_r,
	const float* __restrict__ g_d,
	const float* __restrict__ g_q,
	float* __restrict__ d_r,
	float* __restrict__ d_g,
	float* __restrict__ d_b,
	float* __restrict__ d_a
)
{
	// Collect up to 4 elements and perform partial dot-products
	float4 sums = initialReduction(n, g_r, g_d, g_q);
    
	// Reduce the partial results in this block
	sums = blockReduceSum(sums);

    // Write result for this block to global memory
    if (threadIdx.x == 0)
    {
		d_r[blockIdx.x] = sums.x;
		d_g[blockIdx.x] = sums.y;
		d_b[blockIdx.x] = sums.z;
		d_a[blockIdx.x] = sums.w;
	}
}

extern "C"
__global__ void CGComputeReductionShuffleContinue
(	
	unsigned int n,
	const float* __restrict__ in_d_r,
	const float* __restrict__ in_d_g,
	const float* __restrict__ in_d_b,
	const float* __restrict__ in_d_a,
	float* __restrict__ out_d_r,
	float* __restrict__ out_d_g,
	float* __restrict__ out_d_b,
	float* __restrict__ out_d_a
)
{
	// Collect up to 4 elements
	float4 sums = reduction(n, in_d_r, in_d_g, in_d_b, in_d_a);
    
	// Reduce the partial results in this block
	sums = blockReduceSum(sums);
	
    // Write result for this block to global memory
    if (threadIdx.x == 0)
    {
		out_d_r[blockIdx.x] = sums.x;
		out_d_g[blockIdx.x] = sums.y;
		out_d_b[blockIdx.x] = sums.z;
		out_d_a[blockIdx.x] = sums.w;
	}
}

extern "C"
__global__ void CGComputeReductionShuffleAtomics
(
	unsigned int n,
	const float* __restrict__ g_r,
	const float* __restrict__ g_d,
	const float* __restrict__ g_q,
	float* __restrict__ d_r,
	float* __restrict__ d_g,
	float* __restrict__ d_b,
	float* __restrict__ d_a
)
{
	// Collect up to 4 elements and perform partial dot-products
	float4 sums = initialReduction(n, g_r, g_d, g_q);
    
	// Reduce the partial results in this block
	sums = blockReduceSum(sums);

    // Write result for this block to global memory
    if (threadIdx.x == 0)
    {
		atomicAdd(d_r, sums.x);
		atomicAdd(d_g, sums.y);
		atomicAdd(d_b, sums.z);
		atomicAdd(d_a, sums.w);
	}
}

extern "C"
__global__ void CGUpdateVectorsEx
(
	const unsigned int n,
	const float* __restrict__ d_r_ptr,
	const float* __restrict__ d_g_ptr,
	const float* __restrict__ d_b_ptr,
	const float* __restrict__ d_a_ptr,
	float* __restrict__ vX,
	float* __restrict__ vD,
	float* __restrict__ vQ,
	float* __restrict__ vR
)
{
	// Declare the necessary variables
	float x0, x1, x2, x3;
	float d0, d1, d2, d3;
	float r0, r1, r2, r3;
	float q0, q1, q2, q3;

	unsigned int i0, i1, i2, i3;

	// Compute the index for the first element to fetch
	i0 = 4*blockIdx.x*blockDim.x + threadIdx.x;

	// Fetch the computed dot products
	float d_r = *d_r_ptr;
	float d_g = *d_g_ptr;
	float d_b = *d_b_ptr;
	float d_a = *d_a_ptr;

	// Compute the second index
	i1 = i0 + blockDim.x;

	// Fetch the first elements
	if (i0 < n)
	{
		x0 = vX[i0];
		d0 = vD[i0];
		r0 = vR[i0];
		q0 = vQ[i0];
	}

	// Compute the update constants
	float alpha = 0.0f;
	if (abs(d_g) > 0.0f)
		alpha = d_r / d_g;

	float beta = d_r - (2.0f * alpha) * d_b + (alpha * alpha) * d_a;
	if (abs(d_r) > 0.0f)
		beta = beta / d_r;

	// Fetch the second elements
	if (i1 < n)
	{
		x1 = vX[i1];
		d1 = vD[i1];
		r1 = vR[i1];
		q1 = vQ[i1];
	}

	// Compute the third index
	i2 = i1 + blockDim.x;

	// Update the first elements
	if (i0 < n)
	{
		x0 += alpha * d0;
		r0 -= alpha * q0;
		d0 = r0 + beta * d0;
	}

	// Compute the fourth index
	i3 = i2 + blockDim.x;

	// Fetch the third elements
	if (i2 < n)
	{
		x2 = vX[i2];
		d2 = vD[i2];
		r2 = vR[i2];
		q2 = vQ[i2];
	}

	// Store the first elements
	if (i0 < n)
	{
		vX[i0] = x0;
		vR[i0] = r0;
		vD[i0] = d0;
	}

	// Update the second elements
	if (i1 < n)
	{
		x1 += alpha * d1;
		r1 -= alpha * q1;
		d1 = r1 + beta * d1;
	}
	
	// Fetch the fourth elements
	if (i3 < n)
	{
		x3 = vX[i3];
		d3 = vD[i3];
		r3 = vR[i3];
		q3 = vQ[i3];
	}

	// Store the second elements
	if (i1 < n)
	{
		vX[i1] = x1;
		vR[i1] = r1;
		vD[i1] = d1;
	}

	// Update the third elements
	if (i2 < n)
	{
		x2 += alpha * d2;
		r2 -= alpha * q2;
		d2 = r2 + beta * d2;
	}

	// Update the fourth elements
	if (i3 < n)
	{
		x3 += alpha * d3;
		r3 -= alpha * q3;
		d3 = r3 + beta * d3;
	}

	// Store the third elements
	if (i2 < n)
	{
		vX[i2] = x2;
		vR[i2] = r2;
		vD[i2] = d2;
	}

	// Store the fourth elements
	if (i3 < n)
	{
		vX[i3] = x3;
		vR[i3] = r3;
		vD[i3] = d3;
	}
}
