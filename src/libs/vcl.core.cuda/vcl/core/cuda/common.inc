#ifndef VCL_COMPUTE_CUDA_COMMON_H
#define VCL_COMPUTE_CUDA_COMMON_H

// Control architecture specifics
#if __CUDA_ARCH__ >= 200
#	define VCL_CUDA_NR_SM_BANKS 32
#elif __CUDA_ARCH__ < 200
#	define VCL_CUDA_NR_SM_BANKS 16
#endif /* __CUDA_ARCH__ >= 200 */

// Shared Memory
// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// Specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

// Bank strided shared memory access
#define VCL_CUDA_STRIDED_SM_ACCESS(bank_ptr, idx) (bank_ptr)[VCL_CUDA_NR_SM_BANKS*idx]

// Translate thread idx to Shared Memory address
__device__ inline int tidtosmaddr(int tid)
{
	int n = tid >> 4;
	int m = tid & 15;
	int addr = n * 17 + m;
	return addr;
}

// Math functions
__device__ inline float3 operator - (float3 a, float3 b)
{
	return make_float3(a.x - b.x,
					   a.y - b.y,
					   a.z - b.z);
}

__device__ inline float3 operator / (float3 a, float b)
{
	return make_float3(a.x / b,
					   a.y / b,
					   a.z / b);
}

__device__ inline float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y,
					   a.z * b.x - a.x * b.z,
					   a.x * b.y - a.y * b.x);
}

#endif /* VCL_COMPUTE_CUDA_COMMON_H */
