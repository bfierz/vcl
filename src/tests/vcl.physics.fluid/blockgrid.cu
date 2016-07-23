
#if !(defined(__cplusplus) && defined(__CUDACC__))
#	error "Compile with CUDA compiler"
#endif

#include <vcl/core/cuda/common.inc>
#include <vcl/physics/fluid/cuda/centergrid.inc>

extern "C"
__global__ void BlockLinearGridIndex
(
	int4* indices,
	const dim3 res
)
{
	// Current index
	GridBlockedIdx3D<8> idx(res.x, res.y, res.z);

	const int i = idx.x();
	const int j = idx.y();
	const int k = idx.z();

	const int index = idx();

	indices[index].x = i;
	indices[index].y = j;
	indices[index].z = k;
	indices[index].w = -1;
}
