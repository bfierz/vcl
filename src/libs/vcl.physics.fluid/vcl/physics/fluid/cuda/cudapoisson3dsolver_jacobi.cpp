#include <vcl/physics/fluid/cuda/cudapoisson3dsolver_jacobi.h>

// C++ standard library
#include <iostream>

// VCL library
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/device.h>
#include <vcl/compute/cuda/module.h>

#include <vcl/physics/fluid/cuda/cudacentergrid.h>

#ifdef VCL_CUDA_SUPPORT
extern uint32_t Poisson3dSolverJacobiCudaModule[];
extern size_t Poisson3dSolverJacobiCudaModuleSize;

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	CenterGrid3DPoissonJacobiCtx::CenterGrid3DPoissonJacobiCtx
	(
		ref_ptr<Vcl::Compute::Cuda::Context> ctx,
		ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue,
		Eigen::Vector3i dim
	)
	: JacobiContext()
	, _context(ctx)
	, _dim(dim)
	, _queue(queue)
	{
		// Load the module
		_jacobiModule = static_pointer_cast<Compute::Cuda::Module>(ctx->createModuleFromSource((int8_t*) Poisson3dSolverJacobiCudaModule, Poisson3dSolverJacobiCudaModuleSize * sizeof(uint32_t)));

		if (_jacobiModule)
		{
			// Load the related kernel functions
			_jacobiUpdateSolution = static_pointer_cast<Compute::Cuda::Kernel>(_jacobiModule->kernel("UpdateSolution"));
		}
	}
	CenterGrid3DPoissonJacobiCtx::~CenterGrid3DPoissonJacobiCtx()
	{
		context()->release(_jacobiModule);
	}

	void CenterGrid3DPoissonJacobiCtx::setData
	(
		std::array<ref_ptr<Compute::Buffer>, 4> laplacian,
		ref_ptr<Compute::Buffer> obstacles,
		ref_ptr<Compute::Buffer> pressure,
		ref_ptr<Compute::Buffer> divergence,
		Eigen::Vector3i dim
	)
	{
		using Compute::Cuda::Buffer;

		VclRequire(dim.x() >= 16 && dim.x() <= 128, "Dimension of grid are valid.");
		VclRequire(dim.y() >= 16 && dim.y() <= 128, "Dimension of grid are valid.");
		VclRequire(dim.z() >= 16 && dim.z() <= 128, "Dimension of grid are valid.");
		VclRequire(dynamic_pointer_cast<Buffer>(obstacles), "obstacles is CUDA buffer");
		VclRequire(dynamic_pointer_cast<Buffer>(pressure), "pressure is CUDA buffer");
		VclRequire(dynamic_pointer_cast<Buffer>(divergence), "divergence is CUDA buffer");

		_laplacian = laplacian;
		_obstacles  = static_pointer_cast<Buffer>(obstacles);
		_pressure   = static_pointer_cast<Buffer>(pressure);
		_divergence = static_pointer_cast<Buffer>(divergence);
		_dim = dim;

		
		if (!_solution)
		{
			_solution = static_pointer_cast<Buffer>(context()->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, this->size()*sizeof(float)));
		}
		else
		{
			_solution->resize(this->size()*sizeof(float));
		}
	}

	int CenterGrid3DPoissonJacobiCtx::size() const
	{
		return _dim.x()*_dim.y()*_dim.z();
	}

	void CenterGrid3DPoissonJacobiCtx::precompute()
	{

	}

	void CenterGrid3DPoissonJacobiCtx::updateSolution()
	{
		using Compute::Cuda::Buffer;
		using Compute::Cuda::CommandQueue;

		// Initialize the solver
		_queue->copy(_pressure, _divergence);

		// Fetch the internal solver buffers
		auto& laplacian0 = static_cast<Buffer&>(*_laplacian[0]);
		auto& laplacian1 = static_cast<Buffer&>(*_laplacian[1]);
		auto& laplacian2 = static_cast<Buffer&>(*_laplacian[2]);
		auto& laplacian3 = static_cast<Buffer&>(*_laplacian[3]);
		auto& pressure   = *_pressure;
		auto& divergence = *_divergence;
		auto& obstacles  = *_obstacles;
		auto& sol        = *_solution;

		// Dimension of the grid
		unsigned int x = _dim.x();
		unsigned int y = _dim.y();
		unsigned int z = _dim.z();
		dim3 dimension(x, y, z);

		// Kernel configuration
		const int N = 2;
		const dim3 block_size(16, 2, 2);
		dim3 grid_size(x / 16, y / (2 * N), z / 2);

		_jacobiUpdateSolution->run
		(
			static_cast<CommandQueue&>(*_queue),
			grid_size,
			block_size,
			0,
			laplacian0,
			laplacian1,
			laplacian2,
			laplacian3,
			pressure,
			divergence,
			obstacles,
			sol,
			dimension
		);

//#define VCL_PHYSICS_FLUID_CUDA_JACOBI_VERIFY
#ifdef VCL_PHYSICS_FLUID_CUDA_JACOBI_VERIFY
		// Compare against CPU implementation
		unsigned int X = x;
		unsigned int Y = y;
		unsigned int Z = z;
		
		std::vector<float> a(x*y*z, 0.0f);
		std::vector<float> b(x*y*z, 0.0f);
		std::vector<float> c(x*y*z, 0.0f);

		std::vector<float> Acenter(x*y*z, 0.0f);
		std::vector<float> Ax(x*y*z, 0.0f);
		std::vector<float> Ay(x*y*z, 0.0f);
		std::vector<float> Az(x*y*z, 0.0f);
		std::vector<float> skip(x*y*z, 0.0f);
		
		cuMemcpyDtoH(a.data(), pressure.devicePtr(), a.size() * sizeof(float));
		cuMemcpyDtoH(b.data(), divergence.devicePtr(), b.size() * sizeof(float));
		cuMemcpyDtoH(c.data(), sol.devicePtr(), c.size() * sizeof(float));

		cuMemcpyDtoH(Acenter.data(), laplacian0.devicePtr(), Acenter.size() * sizeof(float));
		cuMemcpyDtoH(Ax.data(), laplacian1.devicePtr(), Ax.size() * sizeof(float));
		cuMemcpyDtoH(Ay.data(), laplacian3.devicePtr(), Ay.size() * sizeof(float));
		cuMemcpyDtoH(Az.data(), laplacian2.devicePtr(), Az.size() * sizeof(float));
		cuMemcpyDtoH(skip.data(), obstacles.devicePtr(), skip.size() * sizeof(float));

		//auto Acenter = (float*) laplacian0.map();
		//auto Ax = (float*) laplacian1.map();
		//auto Ay = (float*) laplacian3.map();
		//auto Az = (float*) laplacian2.map();
		
		auto field = a.data();
		auto rhs = b.data();
		auto result = c.data();
		//auto skip = (float*) obstacles.map();
		
		// x^{n+1} = D^-1 (b - R x^{n})
		//                -------------
		//                      q
		float error_L1 = 0.0f;
		size_t index = X*Y + X + 1;
		for (size_t sz = 1; sz < Z - 1; sz++, index += 2 * X)
		{
			for (size_t sy = 1; sy < Y - 1; sy++, index += 2)
			{
				for (size_t sx = 1; sx < X - 1; sx++, index++)
				{
					float q =
						field[index - 1] * Ax[index] +
						field[index + 1] * Ax[index - 1] +
						field[index - X] * Ay[index] +
						field[index + X] * Ay[index - X] +
						field[index - X*Y] * Az[index] +
						field[index + X*Y] * Az[index - X*Y];
		
					q = 1.0f / Acenter[index] * (rhs[index] - q);
		
					q = (skip[index]) ? 0.0f : q;
		
					error_L1 += abs(result[index] - q);
				}
			}
		}
		
		std::cout << "L1: " << error_L1 << std::endl;
#endif // VCL_PHYSICS_FLUID_CUDA_JACOBI_VERIFY

		_queue->copy(_pressure, _solution);
	}

	double CenterGrid3DPoissonJacobiCtx::computeError()
	{
		return 0;
	}

	void CenterGrid3DPoissonJacobiCtx::finish(double* residual)
	{
	}
}}}}
#endif /* VCL_CUDA_SUPPORT */
