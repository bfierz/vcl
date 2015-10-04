#ifndef VCL_COMPUTE_CUDA_GRID_CUH
#define VCL_COMPUTE_CUDA_GRID_CUH

template<int BlockSize>
class GridIdx3D
{
public:
	__device__ inline GridIdx3D(int x, int y, int z)
	: _x(x)
	, _y(y)
	, _z(z)
	{
	}

	__device__ inline int operator() () const
	{
#ifdef VCL_CUDA_BLOCK_GRID
		const int blockSize = BlockSize*BlockSize*BlockSize;

		int W = BlockSize / blockDim.x;
		int H = BlockSize / blockDim.y;
		int D = BlockSize / blockDim.z;
		
		int X = blockIdx.x / W;
		int Y = blockIdx.y / H;
		int Z = blockIdx.z / D;

		// Compute the index of the grid-block
		int bIdx = X + Y*BlockSize + Z*BlockSize*BlockSize;

		// Compute the index within the block
		int x = blockDim.x * (BlockSize % blockDim.x) + threadIdx.x;
		int y = blockDim.y * (BlockSize % blockDim.y) + threadIdx.y;
		int z = blockDim.z * (BlockSize % blockDim.z) + threadIdx.z;

		return blockSize*bIdx + x + y*BlockSize + z*BlockSize*BlockSize;
#else
		return x() + y()*X() + z()*X()*Y();
#endif /* VCL_CUDA_BLOCK_GRID */
	}

	__device__ inline int operator() (int i, int j, int k) const
	{
#ifdef VCL_CUDA_BLOCK_GRID
		const int blockSize = BlockSize*BlockSize*BlockSize;
		
		int X = i / BlockSize;
		int Y = j / BlockSize;
		int Z = k / BlockSize;

		// Compute the index of the grid-block
		int bIdx = X + Y*BlockSize + Z*BlockSize*BlockSize;

		// Compute the index within the block
		int x = threadIdx.x % BlockSize;
		int y = threadIdx.y % BlockSize;
		int z = threadIdx.z % BlockSize;

		return blockSize*bIdx + x + y*BlockSize + z*BlockSize*BlockSize;
#else
		return i + j*X() + k*X()*Y();
#endif /* VCL_CUDA_BLOCK_GRID */
	}

	__device__ inline int x() const
	{
#ifdef VCL_CUDA_BLOCK_GRID
		int W = BlockSize / blockDim.x;
		int X = blockIdx.x / W;
		int x = blockDim.x * (BlockSize % blockDim.x) + threadIdx.x;

		return x + X*BlockSize;
#else
		return 1 * blockIdx.x*blockDim.x + 1 * threadIdx.x;
#endif /* VCL_CUDA_BLOCK_GRID */
	}

	__device__ inline int y() const
	{
#ifdef VCL_CUDA_BLOCK_GRID
		int H = BlockSize / blockDim.y;
		int Y = blockIdx.y / H;
		int y = blockDim.y * (BlockSize % blockDim.y) + threadIdx.y;

		return y + Y*BlockSize;
#else
		return 1 * blockIdx.y*blockDim.y + 1 * threadIdx.y;
#endif /* VCL_CUDA_BLOCK_GRID */
	}

	__device__ inline int z() const
	{
#ifdef VCL_CUDA_BLOCK_GRID
		int D = BlockSize / blockDim.z;
		int Z = blockIdx.z / D;
		int z = blockDim.z * (BlockSize % blockDim.z) + threadIdx.z;

		return z * Z*BlockSize;
#else
		return 1 * blockIdx.z*blockDim.z + 1 * threadIdx.z;
#endif /* VCL_CUDA_BLOCK_GRID */
	}

	__device__ inline int X() const
	{
		return _x;
	}

	__device__ inline int Y() const
	{
		return _y;
	}

	__device__ inline int Z() const
	{
		return _z;
	}

private:
	int _x;
	int _y;
	int _z;
};

template<int BlockSize>
class GridBlockedIdx3D
{
public:
	__device__ inline GridBlockedIdx3D(int x, int y, int z)
	: _x(x)
	, _y(y)
	, _z(z)
	{
	}

	__device__ inline int operator() () const
	{
		const int blockSize = BlockSize*BlockSize*BlockSize;

		//int W = BlockSize / blockDim.x;
		//int H = BlockSize / blockDim.y;
		//int D = BlockSize / blockDim.z;

		//int X = blockIdx.x / W;
		//int Y = blockIdx.y / H;
		//int Z = blockIdx.z / D;

		int X = blockIdx.x;
		int Y = blockIdx.y;
		int Z = blockIdx.z / 2;

		// Compute the index of the grid-block
		int bIdx = X + Y*BlockSize + Z*BlockSize*BlockSize;

		// Compute the index within the block
		//int x = blockDim.x * (blockIdx.x % W) + threadIdx.x;
		//int y = blockDim.y * (blockIdx.y % H) + threadIdx.y;
		//int z = blockDim.z * (blockIdx.z % D) + threadIdx.z;
		int x = threadIdx.x;
		int y = threadIdx.y;
		int z = blockDim.z * (blockIdx.z % 2) + threadIdx.z;

		return blockSize*bIdx + x + y*BlockSize + z*BlockSize*BlockSize;
	}

	__device__ inline int operator() (int i, int j, int k) const
	{
		const int blockSize = BlockSize*BlockSize*BlockSize;

		int X = i >> 3;// / BlockSize;
		int Y = j >> 3;// / BlockSize;
		int Z = k >> 3;// / BlockSize;

		// Compute the index of the grid-block
		int bIdx = X + Y*BlockSize + Z*BlockSize*BlockSize;

		// Compute the index within the block
		int x = i & 7;//% BlockSize;
		int y = j & 7;//% BlockSize;
		int z = k & 7;//% BlockSize;

		return blockSize*bIdx + x + y*BlockSize + z*BlockSize*BlockSize;
	}

	__device__ inline int x() const
	{
		//int W = BlockSize / blockDim.x;
		//int X = blockIdx.x / W;
		//int x = blockDim.x * (blockIdx.x % W) + threadIdx.x;

		int X = blockIdx.x;
		int x = threadIdx.x;

		return x + X*BlockSize;
	}

	__device__ inline int y() const
	{
		//int H = BlockSize / blockDim.y;
		//int Y = blockIdx.y / H;
		//int y = blockDim.y * (blockIdx.y % H) + threadIdx.y;

		int Y = blockIdx.y;
		int y = threadIdx.y;

		return y + Y*BlockSize;
	}

	__device__ inline int z() const
	{
		//int D = BlockSize / blockDim.z;
		//int Z = blockIdx.z / D;
		//int z = blockDim.z * (blockIdx.z % D) + threadIdx.z;

		int Z = blockIdx.z / 2;
		int z = blockDim.z * (blockIdx.z % 2) + threadIdx.z;

		return z * Z*BlockSize;
	}

	__device__ inline int X() const
	{
		return _x;
	}

	__device__ inline int Y() const
	{
		return _y;
	}

	__device__ inline int Z() const
	{
		return _z;
	}

private:
	int _x;
	int _y;
	int _z;
};

#endif /* VCL_COMPUTE_CUDA_GRID_CUH */
