
#if !(defined(__cplusplus) && defined(__CUDACC__))
#	error "Compile with CUDA compiler"
#endif

#include <vcl/core/cuda/common.inc>
#include <vcl/physics/fluid/cuda/centergrid.inc>

typedef unsigned int uint;

extern "C"
__global__ void ComputeVorticity
(
	const float* __restrict__ VelocityX,
	const float* __restrict__ VelocityY,
	const float* __restrict__ VelocityZ,
	const uint*  __restrict__ Obstacle,
	float* __restrict__ VorticityX,
	float* __restrict__ VorticityY,
	float* __restrict__ VorticityZ,
	float* __restrict__ Vorticity,
	float halfInvCellSize,
	dim3 res
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
		uint  up    = Obstacle[index + res.x] ? index : index + res.x;
		uint  down  = Obstacle[index - res.x] ? index : index - res.x;
		float dy    = (up == index || down == index) ? 2.0f*halfInvCellSize : halfInvCellSize;

		uint  out   = Obstacle[index + res.x*res.y] ? index : index + res.x*res.y;
		uint  in    = Obstacle[index - res.x*res.y] ? index : index - res.x*res.y;
		float dz    = (out == index || in == index) ? 2.0f*halfInvCellSize : halfInvCellSize;

		uint  right = Obstacle[index + 1] ? index : index + 1;
		uint  left  = Obstacle[index - 1] ? index : index - 1;
		float dx    = (right == index || right == index) ? 2.0f*halfInvCellSize : halfInvCellSize;

		float vX = (VelocityZ[up]    - VelocityZ[down]) * dy + (-VelocityY[out]   + VelocityY[in])   * dz;
		float vY = (VelocityX[out]   - VelocityX[in])   * dz + (-VelocityZ[right] + VelocityZ[left]) * dx;
		float vZ = (VelocityY[right] - VelocityY[left]) * dx + (-VelocityX[up]    + VelocityX[down]) * dy;
		VorticityX[index] = vX;
		VorticityY[index] = vY;
		VorticityZ[index] = vZ;

		Vorticity[index] = sqrtf(vX * vX + vY * vY + vZ * vZ);
	}
}

extern "C"
__global__ void AddVorticity
(
	const float* __restrict__ VorticityX,
	const float* __restrict__ VorticityY,
	const float* __restrict__ VorticityZ,
	const float* __restrict__ Vorticity,
	const uint*  __restrict__ Obstacle,
	float* __restrict__ ForceX,
	float* __restrict__ ForceY,
	float* __restrict__ ForceZ,
	float vorticityEps,
	float cellSize,
	float halfInvCellSize,
	dim3 res
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
		0 < z && z < res.z - 1 &&
		!Obstacle[index]        )
	{
		uint  up    = Obstacle[index + res.x] ? index : index + res.x;
		uint  down  = Obstacle[index - res.x] ? index : index - res.x;
		float dy    = (up == index || down == index) ? 2.0f*halfInvCellSize : halfInvCellSize;

		uint  out   = Obstacle[index + res.x*res.y] ? index : index + res.x*res.y;
		uint  in    = Obstacle[index - res.x*res.y] ? index : index - res.x*res.y;
		float dz    = (out == index || in == index) ? 2.0f*halfInvCellSize : halfInvCellSize;

		uint  right = Obstacle[index + 1] ? index : index + 1;
		uint  left  = Obstacle[index - 1] ? index : index - 1;
		float dx    = (right == index || right == index) ? 2.0f*halfInvCellSize : halfInvCellSize;

		float3 N;
		N.x = (Vorticity[right] - Vorticity[left]) * dx; 
		N.y = (Vorticity[up]    - Vorticity[down]) * dy;
		N.z = (Vorticity[out]   - Vorticity[in])   * dz;

		float magnitude = sqrt(N.x*N.x + N.y*N.y + N.z*N.z);
		if (magnitude > 0.0f)
		{
			 magnitude = 1.0f / magnitude;
			 N.x *= magnitude;
			 N.y *= magnitude;
			 N.z *= magnitude;

			ForceX[index] += (N.y * VorticityZ[index] - N.z * VorticityY[index]) * cellSize * vorticityEps;
			ForceY[index] -= (N.x * VorticityZ[index] - N.z * VorticityX[index]) * cellSize * vorticityEps;
			ForceZ[index] += (N.x * VorticityY[index] - N.y * VorticityX[index]) * cellSize * vorticityEps;
		}
	}
}
