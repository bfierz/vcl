
// McAdams SVD library
#define USE_SCALAR_IMPLEMENTATION
#define HAS_RSQRT

// #define USE_ACCURATE_RSQRT_IN_JACOBI_CONJUGATION
// #define PERFORM_STRICT_QUATERNION_RENORMALIZATION
// #define PRINT_DEBUGGING_OUTPUT

#define COMPUTE_V_AS_MATRIX
//#define COMPUTE_V_AS_QUATERNION
#define COMPUTE_U_AS_MATRIX
//#define COMPUTE_U_AS_QUATERNION

#include <vcl/math/mcadams/Singular_Value_Decomposition_Preamble.hpp>

float load3(__global const float* data, int n, int idx, int i)
{
	return data[n*i + idx];
}
float load3x3(__global const float* data, int n, int idx, int i, int j)
{
	return data[n*(j * 3 + i) + idx];
}

void store3(__global float* data, int n, int idx, int i, float value)
{
	data[n*i + idx] = value;
}
void store4(__global float* data, int n, int idx, int i, float value)
{
	data[n*i + idx] = value;
}
void store3x3(__global float* data, int n, int idx, int i, int j, float value)
{
	data[n*(j * 3 + i) + idx] = value;
}

__kernel void JacobiSVD33McAdams
(
	int size,
	int capacity,
	__global const float* __restrict__ A,
	__global float* __restrict__ U,
	__global float* __restrict__ V,
	__global float* __restrict__ S
)
{
#define JACOBI_CONJUGATION_SWEEPS (int) 5

	int idx = (int) get_global_id(0);

#include <vcl/math/mcadams/Singular_Value_Decomposition_Kernel_Declarations.hpp>

	ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = load3x3(A, capacity, idx, 0, 0);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = load3x3(A, capacity, idx, 1, 0);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = load3x3(A, capacity, idx, 2, 0);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = load3x3(A, capacity, idx, 0, 1);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = load3x3(A, capacity, idx, 1, 1);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = load3x3(A, capacity, idx, 2, 1);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = load3x3(A, capacity, idx, 0, 2);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = load3x3(A, capacity, idx, 1, 2);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = load3x3(A, capacity, idx, 2, 2);)

#include <vcl/math/mcadams/Singular_Value_Decomposition_Main_Kernel_Body.hpp>

#ifdef COMPUTE_U_AS_MATRIX
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(U, capacity, idx, 0, 0, Su11.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(U, capacity, idx, 1, 0, Su21.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(U, capacity, idx, 2, 0, Su31.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(U, capacity, idx, 0, 1, Su12.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(U, capacity, idx, 1, 1, Su22.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(U, capacity, idx, 2, 1, Su32.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(U, capacity, idx, 0, 2, Su13.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(U, capacity, idx, 1, 2, Su23.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(U, capacity, idx, 2, 2, Su33.f);)
#endif

#ifdef COMPUTE_U_AS_QUATERNION
	ENABLE_SCALAR_IMPLEMENTATION(store4(U, capacity, idx, 0, Squs.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store4(U, capacity, idx, 1, Squvx.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store4(U, capacity, idx, 2, Squvy.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store4(U, capacity, idx, 3, Squvz.f);)
#endif

#ifdef COMPUTE_V_AS_MATRIX
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(V, capacity, idx, 0, 0, Sv11.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(V, capacity, idx, 1, 0, Sv21.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(V, capacity, idx, 2, 0, Sv31.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(V, capacity, idx, 0, 1, Sv12.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(V, capacity, idx, 1, 1, Sv22.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(V, capacity, idx, 2, 1, Sv32.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(V, capacity, idx, 0, 2, Sv13.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(V, capacity, idx, 1, 2, Sv23.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3x3(V, capacity, idx, 2, 2, Sv33.f);)
#endif

#ifdef COMPUTE_V_AS_QUATERNION
	ENABLE_SCALAR_IMPLEMENTATION(store4(V, capacity, idx, 0, Sqvs.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store4(V, capacity, idx, 1, Sqvvx.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store4(V, capacity, idx, 2, Sqvvy.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store4(V, capacity, idx, 3, Sqvvz.f);)
#endif

	ENABLE_SCALAR_IMPLEMENTATION(store3(S, capacity, idx, 0, Sa11.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3(S, capacity, idx, 1, Sa22.f);)
	ENABLE_SCALAR_IMPLEMENTATION(store3(S, capacity, idx, 2, Sa33.f);)
}
