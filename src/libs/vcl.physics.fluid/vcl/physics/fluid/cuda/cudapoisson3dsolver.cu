
#if !(defined(__cplusplus) && defined(__CUDACC__))
#	error "Compile with CUDA compiler"
#endif

#include <vcl/core/cuda/common.inc>
#include <vcl/physics/fluid/cuda/centergrid.inc>

//#define USE_SM

/*!
 *	\brief Compute the diverence of a velocity field
 */
extern "C"
__global__ void ComputeDivergence
(
	float* Div,
	float* Pressure,
	const float* Velx,
	const float* Vely,
	const float* Velz,
	const unsigned int* skip,
	const dim3 res,
	const float cellSize
)
{
	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);

	const int i = idx.x();
	const int j = idx.y();
	const int k = idx.z();

	const int index = idx();

	// Initialize divergence
	float div = 0;

	if (0 < i && i < res.x - 1 &&
		0 < j && j < res.y - 1 &&
		0 < k && k < res.z - 1)
	{
		float xright  = Velx[index + 1];
		float xleft   = Velx[index - 1];
		float yup     = Vely[index + res.x];
		float ydown   = Vely[index - res.x];
		float ztop    = Velz[index + res.x*res.y];
		float zbottom = Velz[index - res.x*res.y];

		/*if(mObstacles[index + 1]) xright = mObstaclesVelocityX[index + 1];
		if(mObstacles[index - 1]) xleft  = mObstaclesVelocityX[index - 1];
		if(mObstacles[index + mX]) yup    = mObstaclesVelocityY[index + mX];
		if(mObstacles[index - mX]) ydown  = mObstaclesVelocityY[index - mX];
		if(mObstacles[index + mX*mY]) ztop    = mObstaclesVelocityZ[index + mX*mY];
		if(mObstacles[index - mX*mY]) zbottom = mObstaclesVelocityZ[index - mX*mY];*/

		//if (mObstacles[index + 1]) xright = -mVelocityX[index] + mObstaclesVelocityX[index + 1];
		//if (mObstacles[index - 1]) xleft = -mVelocityX[index] + mObstaclesVelocityX[index - 1];
		//if (mObstacles[index + mX]) yup = -mVelocityY[index] + mObstaclesVelocityY[index + mX];
		//if (mObstacles[index - mX]) ydown = -mVelocityY[index] + mObstaclesVelocityY[index - mX];
		//if (mObstacles[index + mX*mY]) ztop = -mVelocityZ[index] + mObstaclesVelocityZ[index + mX*mY];
		//if (mObstacles[index - mX*mY]) zbottom = -mVelocityZ[index] + mObstaclesVelocityZ[index - mX*mY];

		div = -0.5f * cellSize *
				(
					xright - xleft +
					yup - ydown +
					ztop - zbottom
				);
	}

	if (index < res.x*res.y*res.z)
	{
		// Write the divergence to global memory
		Div[index] = div;

		// Reset pressure for the following solver phase
		//Pressure[index] = 0;
	}
}

extern "C"
__global__ void BuildLHS
(
	float* Ac,
	float* Ax,
	float* Ay,
	float* Az,
	const unsigned int* skip,
	const float heat_const,
	const bool heat,
	const dim3 res
)
{
#ifdef USE_SM
	__shared__ unsigned int obstacles[128];
#endif

	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);

	const int i = idx.x();
	const int j = idx.y();
	const int k = idx.z();

	const int index = idx();


#ifdef USE_SM
	const int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;

	// Load parts of the obstacles into shared memory to avoid non-coalesced global memory reads
	obstacles[tid] = skip[index];
	__syncthreads();
#endif

	// Initialize write-back data
	float a_c = 0.0f;
	float a_x = 0.0f;
	float a_y = 0.0f;
	float a_z = 0.0f;

#ifdef USE_SM
	if (!obstacles[tid])
#else
	if (!skip[index])
#endif
	{
		// Set the matrix to the Poisson stencil in order
#ifdef USE_SM
		if (i < (res.x - 1) && !obstacles[tid + 1])
#else
		if (i < (res.x - 1) && !skip[index + 1])
#endif
		{
			a_c += 1.0;
			a_x = -1.0;
		}
#ifdef USE_SM
		if (i > 0 && !obstacles[tid - 1])
#else
		if (i > 0 && !skip[index - 1])
#endif
		{
			a_c += 1.0;
		}
		if (j < (res.y - 1) && !skip[index + res.x])
		{
			a_c += 1.0;
			a_y = -1.0;
		}
		if (j > 0 && !skip[index - res.x])
		{
			a_c += 1.0;
		}
		if (k < (res.z - 1) && !skip[index + res.x*res.y])
		{
			a_c += 1.0;
			a_z = -1.0;
		}
		if (k > 0 && !skip[index - res.x*res.y])
		{
			a_c += 1.0;
		}
	}

	if (index < res.x*res.y*res.z)
	{
		// check to see if we're solving the heat equation instead
		if (heat)
		{
			Ac[index] = 1.0f + heat_const * a_c;
			Ax[index] = heat_const * a_x;
			Ay[index] = heat_const * a_y;
			Az[index] = heat_const * a_z;
		}
		else
		{
			Ac[index] = a_c;
			Ax[index] = a_x;
			Ay[index] = a_y;
			Az[index] = a_z;
		}
	}
}

/*!
 *	\brief Compute the diverence of a velocity field
 */
extern "C"
__global__ void CorrectVelocities
(
	float* Velx,
	float* Vely,
	float* Velz,
	const float* Pressure,
	const unsigned int* skip,
	const dim3 res,
	const float cellInvSize
)
{
	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);

	const int i = idx.x();
	const int j = idx.y();
	const int k = idx.z();

	const int index = idx();

	if (0 < i && i < res.x - 1 &&
		0 < j && j < res.y - 1 &&
		0 < k && k < res.z - 1)
	{
		Velx[index] -= 0.5f * (Pressure[index + 1] - Pressure[index - 1]) * cellInvSize;
		Vely[index] -= 0.5f * (Pressure[index + res.x] - Pressure[index - res.x]) * cellInvSize;
		Velz[index] -= 0.5f * (Pressure[index + res.x*res.y] - Pressure[index - res.x*res.y]) * cellInvSize;
	}
}

#if __CUDA_ARCH__ >= 350
extern "C"
__global__ void Cg3DPoissonSolve
(
	//float* laplacianCenter,
	//float* laplacianX,
	//float* laplacianY,
	//float* laplacianZ,

	float* pressure,
	//const float* divergence,

	const dim3 res
)
{
	if (threadIdx.x == 0)
	{
		const int max_threads = 128;
		dim3 block_size(res.x, max_threads / res.x, 1);
		dim3 grid_size(res.y / block_size.y, res.z);

		//CgInit<<<grid_size, block_size>>>(pressure, res);
		cudaDeviceSynchronize();
	}

	// Wait until all the sub kernels have completed
	__syncthreads();
}
#endif
