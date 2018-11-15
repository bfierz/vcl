#ifndef VCL_COMPUTE_CUDA_MATH_CU
#define VCL_COMPUTE_CUDA_MATH_CU

template<typename TypeT>
__device__ inline float clamp(TypeT value, TypeT lower, TypeT upper)
{
	return max(min(value, upper), lower);
}

__device__ inline float lerp(const float* field, float x, float y, float z, int xres, int yres, int zres)
{
	// clamp pos to grid boundaries
	x = clamp(x, 0.5f, xres - 1.5f);
	y = clamp(y, 0.5f, yres - 1.5f);
	z = clamp(z, 0.5f, zres - 1.5f);

	// locate neighbors to interpolate
	const int x0 = (int)x;
	const int x1 = x0 + 1;
	const int y0 = (int)y;
	const int y1 = y0 + 1;
	const int z0 = (int)z;
	const int z1 = z0 + 1;

	// get interpolation weights
	const float s1 = x - (float)x0;
	const float s0 = 1.0f - s1;
	const float t1 = y - (float)y0;
	const float t0 = 1.0f - t1;
	const float u1 = z - (float)z0;
	const float u0 = 1.0f - u1;

	const int i000 = x0 + y0 * xres + z0 * xres*yres;
	const int i010 = x0 + y1 * xres + z0 * xres*yres;
	const int i100 = x1 + y0 * xres + z0 * xres*yres;
	const int i110 = x1 + y1 * xres + z0 * xres*yres;
	const int i001 = x0 + y0 * xres + z1 * xres*yres;
	const int i011 = x0 + y1 * xres + z1 * xres*yres;
	const int i101 = x1 + y0 * xres + z1 * xres*yres;
	const int i111 = x1 + y1 * xres + z1 * xres*yres;

	// interpolate (indices could be computed once)
	return ( u0 * (s0 * (t0 * field[i000] +
						 t1 * field[i010]) +
				   s1 * (t0 * field[i100] +
						 t1 * field[i110])) +
			 u1 * (s0 * (t0 * field[i001] +
						 t1 * field[i011]) +
				   s1 * (t0 * field[i101] +
						 t1 * field[i111])) );
}

__device__ inline float3 lerp(const float* field1, const float* field2, const float* field3, float x, float y, float z, int xres, int yres, int zres)
{
	// clamp pos to grid boundaries
	x = clamp(x, 0.5f, xres - 1.5f);
	y = clamp(y, 0.5f, yres - 1.5f);
	z = clamp(z, 0.5f, zres - 1.5f);

	// locate neighbors to interpolate
	const int x0 = (int)x;
	const int x1 = x0 + 1;
	const int y0 = (int)y;
	const int y1 = y0 + 1;
	const int z0 = (int)z;
	const int z1 = z0 + 1;

	// get interpolation weights
	const float s1 = x - (float)x0;
	const float s0 = 1.0f - s1;
	const float t1 = y - (float)y0;
	const float t0 = 1.0f - t1;
	const float u1 = z - (float)z0;
	const float u0 = 1.0f - u1;

	const int i000 = x0 + y0 * xres + z0 * xres*yres;
	const int i010 = x0 + y1 * xres + z0 * xres*yres;
	const int i100 = x1 + y0 * xres + z0 * xres*yres;
	const int i110 = x1 + y1 * xres + z0 * xres*yres;
	const int i001 = x0 + y0 * xres + z1 * xres*yres;
	const int i011 = x0 + y1 * xres + z1 * xres*yres;
	const int i101 = x1 + y0 * xres + z1 * xres*yres;
	const int i111 = x1 + y1 * xres + z1 * xres*yres;

	// interpolate (indices could be computed once)
	return make_float3(
			( u0 * (s0 * (t0 * field1[i000] +
				t1 * field1[i010]) +
				s1 * (t0 * field1[i100] +
				t1 * field1[i110])) +
				u1 * (s0 * (t0 * field1[i001] +
				t1 * field1[i011]) +
				s1 * (t0 * field1[i101] +
				t1 * field1[i111])) ) , 
			( u0 * (s0 * (t0 * field2[i000] +
				t1 * field2[i010]) +
				s1 * (t0 * field2[i100] +
				t1 * field2[i110])) +
				u1 * (s0 * (t0 * field2[i001] +
				t1 * field2[i011]) +
				s1 * (t0 * field2[i101] +
				t1 * field2[i111])) ) , 
			( u0 * (s0 * (t0 * field3[i000] +
				t1 * field3[i010]) +
				s1 * (t0 * field3[i100] +
				t1 * field3[i110])) +
				u1 * (s0 * (t0 * field3[i001] +
				t1 * field3[i011]) +
				s1 * (t0 * field3[i101] +
				t1 * field3[i111])) ) 
			);
}

__device__ inline float4 lerp(const float4* field, float x, float y, float z, int xres, int yres, int zres)
{
	// clamp pos to grid boundaries
	x = clamp(x, 0.5f, xres - 1.5f);
	y = clamp(y, 0.5f, yres - 1.5f);
	z = clamp(z, 0.5f, zres - 1.5f);

	// locate neighbors to interpolate
	const int x0 = (int)x;
	const int x1 = x0 + 1;
	const int y0 = (int)y;
	const int y1 = y0 + 1;
	const int z0 = (int)z;
	const int z1 = z0 + 1;

	// get interpolation weights
	const float s1 = x - (float)x0;
	const float s0 = 1.0f - s1;
	const float t1 = y - (float)y0;
	const float t0 = 1.0f - t1;
	const float u1 = z - (float)z0;
	const float u0 = 1.0f - u1;

	const int i000 = x0 + y0 * xres + z0 * xres*yres;
	const int i010 = x0 + y1 * xres + z0 * xres*yres;
	const int i100 = x1 + y0 * xres + z0 * xres*yres;
	const int i110 = x1 + y1 * xres + z0 * xres*yres;
	const int i001 = x0 + y0 * xres + z1 * xres*yres;
	const int i011 = x0 + y1 * xres + z1 * xres*yres;
	const int i101 = x1 + y0 * xres + z1 * xres*yres;
	const int i111 = x1 + y1 * xres + z1 * xres*yres;

	// interpolate (indices could be computed once)
	return make_float4(
			( u0 * (s0 * (t0 * field[i000].x +
						  t1 * field[i010].x) +
				    s1 * (t0 * field[i100].x +
				          t1 * field[i110].x)) +
			  u1 * (s0 * (t0 * field[i001].x +
				          t1 * field[i011].x) +
				    s1 * (t0 * field[i101].x +
						  t1 * field[i111].x))),

			( u0 * (s0 * (t0 * field[i000].y +
						  t1 * field[i010].y) +
				    s1 * (t0 * field[i100].y +
				          t1 * field[i110].y)) +
			  u1 * (s0 * (t0 * field[i001].y +
				          t1 * field[i011].y) +
				    s1 * (t0 * field[i101].y +
						  t1 * field[i111].y))), 
						 
			( u0 * (s0 * (t0 * field[i000].z +
						  t1 * field[i010].z) +
				    s1 * (t0 * field[i100].z +
				          t1 * field[i110].z)) +
			  u1 * (s0 * (t0 * field[i001].z +
				          t1 * field[i011].z) +
				    s1 * (t0 * field[i101].z +
						  t1 * field[i111].z))), 

			( u0 * (s0 * (t0 * field[i000].w +
						  t1 * field[i010].w) +
				    s1 * (t0 * field[i100].w +
				          t1 * field[i110].w)) +
			  u1 * (s0 * (t0 * field[i001].w +
				          t1 * field[i011].w) +
				    s1 * (t0 * field[i101].w +
						  t1 * field[i111].w)))
			);
}

__device__ inline float frac(float x)
{
	return x - floor(x);
}

__device__ inline float2 lut1D(const float2* field, float t, int size)
{
	// locate neighbors to interpolate
	const float x = t * (float)(size - 1);
	const int x0 = (int)x;
	const int x1 = x0 + 1;

	// get interpolation weights
	const float s1 = x - (float)x0;
	const float s0 = 1.0f - s1;

	// interpolate (indices could be computed once)

	const float2 f0 = field[x0];
	const float2 f1 = field[x1];

	return make_float2( s0 * f0.x + s1 * f1.x,
						s0 * f0.y + s1 * f1.y );
}

__device__ inline
float4 warpReduceSum(float4 val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2) 
	{
		val.x += __shfl_down_sync(0xffffffff, val.x, offset);
		val.y += __shfl_down_sync(0xffffffff, val.y, offset);
		val.z += __shfl_down_sync(0xffffffff, val.z, offset);
		val.w += __shfl_down_sync(0xffffffff, val.w, offset);
	}
	return val;
}

__device__ inline
float4 blockReduceSum(float4 val)
{
	// Shared mem for 32 partial sums
	static __shared__ float4 shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	// Each warp performs partial reduction
	val = warpReduceSum(val);
	if (lane == 0)
		shared[wid] = val;
	__syncthreads();

	// Read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_float4(0, 0, 0, 0);

	// Final reduce within first warp
	if (wid == 0)
		val = warpReduceSum(val);

	return val;
}

#endif // VCL_COMPUTE_CUDA_MATH_CU
