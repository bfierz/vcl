
#if !(defined(__cplusplus) && defined(__CUDACC__))
#	error "Compile with CUDA compiler"
#endif

#include <vcl/core/cuda/common.inc>

//#define USE_SM

extern "C"
__global__ void UpdateSolution
(
	const float* __restrict__ Acenter,
	const float* __restrict__ Ax,
	const float* __restrict__ Ay,
	const float* __restrict__ Az,
	const float* __restrict__ curr,
	const float* __restrict__ div,
	const unsigned int* __restrict__ skip,
	float* __restrict__ next,
	const dim3 res
)
{
	// Iterations per each thread does
	const int N = 2;

	// Store a block of rhs values
	//__shared__ float v[16*6*4];

	int i = 1 * blockIdx.x*blockDim.x + 1 * threadIdx.x;
	int j = N*blockIdx.y*blockDim.y + N*threadIdx.y;
	int k = 1 * blockIdx.z*blockDim.z + 1 * threadIdx.z;

	// Load parts of the data into shared memory to avoid non-coalesced global memory reads
	//int tid = 16 * 6 * (2*threadIdx.z) + 16 * threadIdx.y + threadIdx.x;
	//if (k > 1)
	//{
	//	v[tid]               = curr[index - res.x*res.y];
	//	v[tid + res.x*res.y] = curr[index - res.x*res.y];
	//}
	//
	//__syncthreads();

	if (0 < i && i < res.x - 1 &&
		0 < k && k < res.z - 1)
	{
		int index = i + j*res.x + k*res.x*res.y;

#pragma unroll
		for (int n = 0; n < N; n++)
		{
			if (0 < j && j < res.y - 1)
			{
#ifdef USE_SM
				float result = 
					data[tid - 1] * a_x[tid] +
					data[tid + 1] * a_x[tid - 1] +
					curr[index - res.x] * Ay[index] +
					curr[index + res.x] * Ay[index - res.x] +
					curr[index - res.x*res.y] * Az[index] +
					curr[index + res.x*res.y] * Az[index - res.x*res.y];
#else
				float result =
					curr[index - 1] * Ax[index] +
					curr[index + 1] * Ax[index - 1] +
					curr[index - res.x] * Ay[index] +
					curr[index + res.x] * Ay[index - res.x] +
					curr[index - res.x*res.y] * Az[index] +
					curr[index + res.x*res.y] * Az[index - res.x*res.y];
#endif /* USE_SM */

				result = 1.0f / Acenter[index] * (div[index] - result);
				next[index] = (skip[index]) ? 0.0f : result;
			}

			index += res.x;
			j += 1;
		}
	}
}
