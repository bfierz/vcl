
#if !(defined(__cplusplus) && defined(__CUDACC__))
#	error "Compile with CUDA compiler"
#endif

#include <vcl/core/cuda/math.inc>
#include <vcl/physics/fluid/cuda/centergrid.inc>

extern "C"
__global__ void AdvectSemiLagrange
(
	const float dt,
	const float* __restrict__ velx,
	const float* __restrict__ vely,
	const float* __restrict__ velz,
	const float* __restrict__ src,
	float* __restrict__ dst,
	const dim3 res
)
{
	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);

	const int index = idx();

	// Backtrace
	float xTrace = idx.x() - dt * velx[index];
	float yTrace = idx.y() - dt * vely[index];
	float zTrace = idx.z() - dt * velz[index];

	dst[index] = lerp(src, xTrace, yTrace, zTrace, res.x, res.y, res.z);
}

// Global data
texture<float, 3, cudaReadModeElementType> TexOldField;

extern "C"
__global__ void AdvectSemiLagrangeTex
(
	const float dt,
	const float* velx,
	const float* vely,
	const float* velz,
	float* new_field,
	const dim3 res
)
{
	
	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);

	const int index = idx();

	// backtrace
	float xTrace = idx.x() - dt * velx[index];
	float yTrace = idx.y() - dt * vely[index];
	float zTrace = idx.z() - dt * velz[index];

	// clamp backtrace to grid boundaries
	xTrace = clamp(xTrace, 0.5f, res.x - 1.5f);
	yTrace = clamp(yTrace, 0.5f, res.y - 1.5f);
	zTrace = clamp(zTrace, 0.5f, res.z - 1.5f);

	new_field[index] = tex3D(TexOldField, xTrace + 0.5f, yTrace + 0.5f, zTrace + 0.5f);
}
