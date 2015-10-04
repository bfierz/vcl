
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
	float* Acenter,
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

	float a_center = 0.0f;
	float a_x = 0.0f;
	float a_y = 0.0f;
	float a_z = 0.0f;

	if (0 < i && i < res.x - 1 && 
		0 < j && j < res.y - 1 &&
		0 < k && k < res.z - 1)
	{
#ifdef USE_SM
		float skip_center = obstacles[tid];
		float skip_right = obstacles[tid + 1];
		float skip_left = obstacles[tid - 1];
#else
		float skip_center = skip[index];
		float skip_right = skip[index + 1];
		float skip_left = skip[index - 1];
#endif

		float skip_front = skip[index + res.x];
		float skip_back = skip[index - res.x];
		float skip_up = skip[index + res.x*res.y];
		float skip_down = skip[index - res.x*res.y];

		// if the cell is a variable
		if (!skip_center)
		{
			// check to see if we're solving the heat equation instead
			if (heat)
			{
				a_center = 1.0f;

				// set the matrix to the Poisson stencil in order
				if (!skip_right)
				{
					a_center += heat_const;
					a_x = -heat_const;
				}
				if (!skip_left)
				{
					a_center += heat_const;
				}
				if (!skip_front)
				{
					a_center += heat_const;
					a_y = -heat_const;
				}
				if (!skip_back)
				{
					a_center += heat_const;
				}
				if (!skip_up)
				{
					a_center += heat_const;
					a_z = -heat_const;
				}
				if (!skip_down)
				{
					a_center += heat_const;
				}
			}
			else
			{
				// set the matrix to the Poisson stencil in order
				if (!skip_right)
				{
					a_center += 1.0f;
					a_x = -1.0f;
				}
				if (!skip_left)
				{
					a_center += 1.0f;
				}
				if (!skip_front)
				{
					a_center += 1.0f;
					a_y = -1.0f;
				}
				if (!skip_back)
				{
					a_center += 1.0f;
				}
				if (!skip_up)
				{
					a_center += 1.0f;
					a_z = -1.0f;
				}
				if (!skip_down)
				{
					a_center += 1.0f;
				}
			}
		}
		else
		{
			// if the center cell's an obstacle, zero out the matrix row
		}
	}

	if (index < res.x*res.y*res.z)
	{
		Acenter[index] = a_center;
		Ax[index] = a_x;
		Ay[index] = a_y;
		Az[index] = a_z;
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
