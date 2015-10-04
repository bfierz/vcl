#if !(defined(__cplusplus) && defined(__CUDACC__))
#	error "Compile with CUDA compiler"
#endif

#include <vcl/core/cuda/common.inc>
#include <vcl/physics/fluid/cuda/centergrid.inc>

extern "C" __global__ void cuda_engery_compute_energy_density
(
	const float* velx,
	const float* vely,
	const float* velz,
	float* energy,
	const dim3 res
)
{
	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);

	const int index = idx();

	float vx = velx[index];
	float vy = vely[index];
	float vz = velz[index];
	
	energy[index] = 0.5f * (vx*vx + vy*vy + vz*vz);
}
extern "C" __global__ void cuda_energy_march_obstacles_modify_energy
(
	unsigned int* obstacles,
	float* energy,
	const dim3 res
)
{
	enum OBSTACLE_FLAGS
	{
		EMPTY = 0, 
		MARCHED = 2, 
		RETIRED = 4 
	}; 
	
	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);

	int x = idx.x();
	int y = idx.y();
	int z = idx.z();
	
	if (0 < x && x < res.x - 1 &&
		0 < y && y < res.y - 1 && 
		0 < z && z < res.z - 1	  )
	{
		const int index = idx();

		if (obstacles[index] && obstacles[index] != RETIRED)
		{
			float sum = 0.0f;
			int valid = 0;

			if (!obstacles[index + 1] || obstacles[index + 1] == RETIRED)
			{
				sum += energy[index + 1];
				valid++;
			}
			if (!obstacles[index - 1] || obstacles[index - 1] == RETIRED)
			{
				sum += energy[index - 1];
				valid++;
			}
			if (!obstacles[index + res.x] || obstacles[index + res.x] == RETIRED)
			{
				sum += energy[index + res.x];
				valid++;
			}
			if (!obstacles[index - res.x] || obstacles[index - res.x] == RETIRED)
			{
				sum += energy[index - res.x];
				valid++;
			}
			if (!obstacles[index + res.x*res.y] || obstacles[index + res.x*res.y] == RETIRED)
			{
				sum += energy[index + res.x*res.y];
				valid++;
			}
			if (!obstacles[index - res.x*res.y] || obstacles[index - res.x*res.y] == RETIRED)
			{
				sum += energy[index - res.x*res.y];
				valid++;
			}

			if (valid > 0)
			{
				energy[index] = sum / valid;
				obstacles[index] = MARCHED;
			}
		}
	}
}
extern "C" __global__ void cuda_energy_march_obstacles_update_obstacle
(
	unsigned int* obstacles,
	const dim3 res
)
{
	enum OBSTACLE_FLAGS
	{
		EMPTY = 0, 
		MARCHED = 2, 
		RETIRED = 4 
	}; 
	
	// Current index
	GridIdx3D<8> idx(res.x, res.y, res.z);

	int x = idx.x();
	int y = idx.y();
	int z = idx.z();
	
	if (0 < x && x < res.x - 1 &&
		0 < y && y < res.y - 1 && 
		0 < z && z < res.z - 1	  )
	{
		const int index = idx();

		if (obstacles[index] == MARCHED)
			obstacles[index] = RETIRED;
	}
}

__constant__ float aCoeffs[32] = {
	0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
	-0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
	0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
	0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
};

__device__ inline void cuda_energy_downsample(float* to, const float* from, int i, int n, int stride)
{
	const float* const aCoCenter= &aCoeffs[16];

	float result = 0.0f;
	for (int k = 2 * i - 16; k < 2 * i + 16; k++)
	{ 
		// handle boundary
		float fromval; 
		if(k<0) {
			fromval = from[0];
		} else if(k>n-1) {
			fromval = from[(n-1) * stride];
		} else {
			fromval = from[k * stride]; 
		} 
		
		result += aCoCenter[k - 2*i] * fromval; 
	}
	to[i * stride] = result;
}

extern "C" __global__ void cuda_energy_downsample_x
(
	float* to,
	const float* from,
	const dim3 res
)
{
	// Allocate shared memory for this block
	__shared__ float cache[256];

	int x = threadIdx.x;
	int y = blockIdx.x*blockDim.y+threadIdx.y;
	int z = blockIdx.y*blockDim.z+threadIdx.z;
	
	const int index = y * res.x + z * res.x * res.y;

	cache[threadIdx.y * res.x + threadIdx.x] = from[index + x];
	cache[threadIdx.y * res.x + res.x / 2 + threadIdx.x] = from[index + res.x / 2 + x];
	__syncthreads();

	//if (x < res.x / 2)
	//	cuda_energy_downsample(&to[index], &from[index], x, res.x, 1);

	const float* const aCoCenter = &aCoeffs[16];
	const float* source = &cache[threadIdx.y * res.x];
	float* target = &to[index];

	float left_border = source[0];
	float right_border = source[res.x - 1];

	float result = 0.0f;
	for (int k = 2 * x - 16; k < 2 * x + 16; k++)
	{ 
		// handle boundary
		float fromval;
		if (k < 0) {
			fromval = left_border;
		} else if (k > res.x-1) {
			fromval = right_border;
		} else {
			fromval = source[k]; 
		} 
		
		result += aCoCenter[k - 2*x] * fromval; 
	}
	target[x] = result;
}

extern "C" __global__ void cuda_energy_downsample_y
(
	float* to,
	const float* from,
	const dim3 res
)
{
	int x = threadIdx.x;
	int y = blockIdx.x*blockDim.y+threadIdx.y;
	int z = blockIdx.y*blockDim.z+threadIdx.z;
	
	const int index = x + z * res.x * res.y;
	
	//if (y < res.y / 2)
	//	cuda_energy_downsample(&to[index], &from[index], y, res.y, res.x);

	const float* const aCoCenter = &aCoeffs[16];
	const float* source = &from[index];
	float* target = &to[index];

	int stride = res.x;
	float result = 0.0f;
	for (int k = 2 * y - 16; k < 2 * y + 16; k++)
	{ 
		// handle boundary
		float fromval; 
		if(k<0) {
			fromval = source[0];
		} else if(k>res.y-1) {
			fromval = source[(res.y-1) * stride];
		} else {
			fromval = source[k * stride]; 
		} 
		
		result += aCoCenter[k - 2*y] * fromval; 
	}
	target[y * stride] = result;
}

extern "C" __global__ void cuda_energy_downsample_z
(
	float* to,
	const float* from,
	const dim3 res
)
{
	int x = threadIdx.x;
	int y = blockIdx.x*blockDim.y+threadIdx.y;
	int z = blockIdx.y*blockDim.z+threadIdx.z;
	
	const int index = x + y * res.x;
	
	//if (z < res.z / 2)
	//	cuda_energy_downsample(&to[index], &from[index], z, res.z, res.x * res.y);

	const float* const aCoCenter = &aCoeffs[16];
	const float* source = &from[index];
	float* target = &to[index];

	int stride = res.x * res.y;
	float result = 0.0f;
	for (int k = 2 * z - 16; k < 2 * z + 16; k++)
	{ 
		// handle boundary
		float fromval; 
		if(k<0) {
			fromval = source[0];
		} else if(k>res.z-1) {
			fromval = source[(res.z-1) * stride];
		} else {
			fromval = source[k * stride]; 
		} 
		
		result += aCoCenter[k - 2*z] * fromval; 
	}
	target[z * stride] = result;
}

__constant__ float pCoeffs[4] = { 0.25f, 0.75f, 0.75f, 0.25f };

__device__ void cuda_energy_upsample(float *to, const float *from, int i, int n, int stride)
{
	const float *const pCoCenter = &pCoeffs[2];

	//to[i * stride] = 0;
	float result = 0.0f;
	for (int k = i / 2; k <= i / 2 + 1; k++) {
		float fromval;
		// handle boundary
		if(k>n/2) {
			fromval = from[(n/2) * stride];
		} else {
			fromval = from[k * stride]; 
		} 

		//to[i * stride] += pCoCenter[i - 2 * k] * fromval; 
		result += pCoCenter[i - 2 * k] * fromval;
	}
	to[i * stride] = result;
}

extern "C" __global__ void cuda_energy_upsample_x
(
	float* to,
	const float* from,
	const dim3 res
)
{
	__shared__ float cache[128];

	int x = threadIdx.x;
	int y = blockIdx.x*blockDim.y+threadIdx.y;
	int z = blockIdx.y*blockDim.z+threadIdx.z;
	
	const int index = y * res.x + z * res.x * res.y;

	cache[threadIdx.y * blockDim.x + threadIdx.x] = from[index + x];
	__syncthreads();

	//cuda_energy_upsample(&to[index], &from[index], x, res.x, 1);

	const float *const pCoCenter = &pCoeffs[2];
	const float* source = &cache[threadIdx.y * blockDim.x];
	float* target = &to[index];

	float result = 0.0f;
	for (int k = x / 2; k <= x / 2 + 1; k++) {
		float fromval;
		// handle boundary
		if(k>res.x/2) {
			fromval = source[threadIdx.y * blockDim.x + res.x/2];
		} else {
			fromval = source[k]; 
		} 
 
		result += pCoCenter[x - 2 * k] * fromval;
	}
	target[x] = result;
}

extern "C" __global__ void cuda_energy_upsample_y
(
	float* to,
	const float* from,
	const dim3 res
)
{
	int x = threadIdx.x;
	int y = blockIdx.x*blockDim.y+threadIdx.y;
	int z = blockIdx.y*blockDim.z+threadIdx.z;
	
	const int index = x + z * res.x * res.y;
	cuda_energy_upsample(&to[index], &from[index], y, res.y, res.x);
}

extern "C" __global__ void cuda_energy_upsample_z
(
	float* to,
	const float* from,
	const dim3 res
)
{
	int x = threadIdx.x;
	int y = blockIdx.x*blockDim.y+threadIdx.y;
	int z = blockIdx.y*blockDim.z+threadIdx.z;
	
	const int index = x + y * res.x;
	cuda_energy_upsample(&to[index], &from[index], z, res.z, res.x * res.y);
}

extern "C" __global__ void cuda_energy_decompose
(
	const float* low_freq,
	float* energy,
	const dim3 res
)
{
	int x = threadIdx.x;
	int y = blockIdx.x*blockDim.y+threadIdx.y;
	int z = blockIdx.y*blockDim.z+threadIdx.z;
	
	const int index = x + y * res.x + z * res.x * res.y;
	
	energy[index] -= low_freq[index];
}
