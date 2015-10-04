#include <vcl/physics/fluid/cuda/cudapoisson3dsolver.h>

// VCL
#include <vcl/math/solver/conjugategradients.h>
#include <vcl/math/solver/jacobi.h>
#include <vcl/physics/fluid/cuda/cudacentergrid.h>
#include <vcl/physics/fluid/cuda/cudapoisson3dsolver_jacobi.h>

#ifdef VCL_CUDA_SUPPORT
extern uint32_t Poisson3dSolverCudaModule [];
extern size_t Poisson3dSolverCudaModuleSize;

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	CenterGrid3DPoissonCgCtx::CenterGrid3DPoissonCgCtx
	(
		ref_ptr<Vcl::Compute::Cuda::Context> ctx,
		ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue,
		Eigen::Vector3i dim
	)
	: ConjugateGradientsContext(ctx, nullptr, dim.x()*dim.y()*dim.z())
	, _queue(queue)
	{
		// Load the module
		_cgModule = static_pointer_cast<Compute::Cuda::Module>(ctx->createModuleFromSource((int8_t*) Poisson3dSolverCudaModule, Poisson3dSolverCudaModuleSize));

		if (_cgModule)
		{
			// Load the related kernel functions
			_cgInit     = static_pointer_cast<Compute::Cuda::Kernel>(_cgModule->kernel("ComputeInitialResidual"));
			_cgComputeQ = static_pointer_cast<Compute::Cuda::Kernel>(_cgModule->kernel("ComputeQ"));
		}
	}
	CenterGrid3DPoissonCgCtx::~CenterGrid3DPoissonCgCtx()
	{
		context()->release(_cgModule);
	}

	void CenterGrid3DPoissonCgCtx::setData
	(
		std::array<ref_ptr<Vcl::Compute::Cuda::Buffer>, 4> laplacian,
		ref_ptr<Vcl::Compute::Cuda::Buffer> obstacles,
		ref_ptr<Vcl::Compute::Cuda::Buffer> pressure,
		ref_ptr<Vcl::Compute::Cuda::Buffer> divergence,
		Eigen::Vector3i dim
	)
	{
		Require(dim.x() >= 16 && dim.x() <= 128, "Dimension of grid are valid.");
		Require(dim.y() >= 16 && dim.y() <= 128, "Dimension of grid are valid.");
		Require(dim.z() >= 16 && dim.z() <= 128, "Dimension of grid are valid.");

		_laplacian = laplacian;
		_obstacles = obstacles;
		_pressure = pressure;
		_divergence = divergence;
		_dim = dim;

		// Resize the buffers if necessary
		this->resize(dim.x()*dim.y()*dim.z());

		// Set the solution vector
		setX(_pressure);
	}

	void CenterGrid3DPoissonCgCtx::computeInitialResidual()
	{
		// Fetch the internal solver buffers
		auto& laplacian0 = *_laplacian[0];
		auto& laplacian1 = *_laplacian[1];
		auto& laplacian2 = *_laplacian[2];
		auto& laplacian3 = *_laplacian[3];
		auto& obstacles  = *_obstacles;
		auto& pressure   = *_pressure;
		auto& divergence = *_divergence;

		auto& residual  = *static_pointer_cast<Compute::Cuda::Buffer>(_devResidual);
		auto& direction = *static_pointer_cast<Compute::Cuda::Buffer>(_devDirection);

		// Dimension of the grid
		unsigned int x = _dim.x();
		unsigned int y = _dim.y();
		unsigned int z = _dim.z();
		dim3 dimension(x, y, z);

		// Kernel configuration
		const int N = 2;
		const dim3 block_size(16, 2, 2);
		dim3 grid_size(x / 16, y / (2 * N), z / 2);

		_cgInit->run
		(
			*_queue,
			grid_size,
			block_size,
			0,
			(CUdeviceptr) laplacian0,
			(CUdeviceptr) laplacian1,
			(CUdeviceptr) laplacian2,
			(CUdeviceptr) laplacian3,
			(CUdeviceptr) pressure,
			(CUdeviceptr) divergence,
			(CUdeviceptr) obstacles,
			(CUdeviceptr) residual,
			(CUdeviceptr) direction,
			dimension
		);


//#define VCL_PHYSICS_FLUID_CUDA_CG_INIT_VERIFY
#ifdef VCL_PHYSICS_FLUID_CUDA_CG_INIT_VERIFY
		// Compare against CPU implementation
		unsigned int X = x;
		unsigned int Y = y;
		unsigned int Z = z;
		
		std::vector<float> d(x*y*z, 0.0f);
		
		cuMemcpyDtoH(d.data(), mDevResidual, d.size() * sizeof(float));
		
		auto Acenter = (float*) laplacian0.map();
		auto Ax      = (float*) laplacian1.map();
		auto Ay      = (float*) laplacian3.map();
		auto Az      = (float*) laplacian2.map();
		
		auto field = (float*) pressure.map();
		auto rhs = (float*) divergence.map();
		auto skip = (float*) obstacles.map();
		
		float error_L1 = 0.0f;
		size_t index = X*Y + X + 1;
		for (size_t z = 1; z < Z - 1; z++, index += 2 * X)
		{
			for (size_t y = 1; y < Y - 1; y++, index += 2)
			{
				for (size_t x = 1; x < X - 1; x++, index++)
				{
					float r = rhs[index] -
					(
						field[index] * Acenter[index] +
						field[index - 1] * Ax[index] +
						field[index + 1] * Ax[index - 1] +
						field[index - X] * Ay[index] +
						field[index + X] * Ay[index - X] +
						field[index - X*Y] * Az[index] +
						field[index + X*Y] * Az[index - X*Y]
					);
					r = (skip[index]) ? 0.0f : r;
					d[index] -= r;
		
					error_L1 += abs(d[index]);
				}
			}
		}
		
		std::cout << "L1: " << error_L1 << std::endl;
#endif // VCL_PHYSICS_FLUID_CUDA_CG_INIT_VERIFY
	}

	void CenterGrid3DPoissonCgCtx::computeQ()
	{
		// Fetch the internal solver buffers
		auto& laplacian0 = *_laplacian[0];
		auto& laplacian1 = *_laplacian[1];
		auto& laplacian2 = *_laplacian[2];
		auto& laplacian3 = *_laplacian[3];
		auto& obstacles  = *_obstacles;

		// Dimension of the grid
		unsigned int x = _dim.x();
		unsigned int y = _dim.y();
		unsigned int z = _dim.z();
		dim3 dimension(x, y, z);

		// Kernel configuration
		const int N = 2;
		const dim3 block_size(16, 2, 2);
		dim3 grid_size(x / 16, y / (2 * N), z / 2);

		_cgComputeQ->run
		(
			*_queue,
			grid_size,
			block_size,
			0,
			(CUdeviceptr) laplacian0,
			(CUdeviceptr) laplacian1,
			(CUdeviceptr) laplacian2,
			(CUdeviceptr) laplacian3,
			(CUdeviceptr) _devDirection,
			(CUdeviceptr) obstacles,
			(CUdeviceptr) _devQ,
			dimension
		);


//#define VCL_PHYSICS_FLUID_CUDA_CG_Q_VERIFY
#ifdef VCL_PHYSICS_FLUID_CUDA_CG_Q_VERIFY
		// Compare against CPU implementation
		unsigned int X = x;
		unsigned int Y = y;
		unsigned int Z = z;
		
		std::vector<float> d(x*y*z, 0.0f);
		std::vector<float> f(x*y*z, 0.0f);
		
		cuMemcpyDtoH(d.data(), mDevQ, d.size() * sizeof(float));
		cuMemcpyDtoH(f.data(), mDevDirection, f.size() * sizeof(float));
		
		auto Acenter = (float*) laplacian0.map();
		auto Ax = (float*) laplacian1.map();
		auto Ay = (float*) laplacian3.map();
		auto Az = (float*) laplacian2.map();
		
		auto field = f.data();
		auto skip = (float*) obstacles.map();
		
		float error_L1 = 0.0f;
		size_t index = X*Y + X + 1;
		for (size_t z = 1; z < Z - 1; z++, index += 2 * X)
		{
			for (size_t y = 1; y < Y - 1; y++, index += 2)
			{
				for (size_t x = 1; x < X - 1; x++, index++)
				{
					float q =
						field[index] * Acenter[index] +
						field[index - 1] * Ax[index] +
						field[index + 1] * Ax[index - 1] +
						field[index - X] * Ay[index] +
						field[index + X] * Ay[index - X] +
						field[index - X*Y] * Az[index] +
						field[index + X*Y] * Az[index - X*Y];
					q = (skip[index]) ? 0.0f : q;
					d[index] -= q;
		
					error_L1 += abs(d[index]);
				}
			}
		}
		
		std::cout << "L1: " << error_L1 << std::endl;
#endif // VCL_PHYSICS_FLUID_CUDA_CG_Q_VERIFY
	}

	CenterGrid3DPoissonSolver::CenterGrid3DPoissonSolver
	(
		ref_ptr<Vcl::Compute::Cuda::Context> ctx,
		ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue
	)
	: Fluid::CenterGrid3DPoissonSolver()
	, _ownerCtx(ctx)
	, _queue(queue)
	{
		Require(ctx, "Context is set");

		// Is dynamic parallelism supported
		//_supportsFullDeviceSolver = ctx->supports(Vcl::Compute::Cuda::Feature::DynamicParallelism);
		
		// Load the module
		_module = static_pointer_cast<Compute::Cuda::Module>(ctx->createModuleFromSource((int8_t*) Poisson3dSolverCudaModule, Poisson3dSolverCudaModuleSize));

		if (_module)
		{
			if (_supportsFullDeviceSolver)
			{
				_fullDeviceSolver = static_pointer_cast<Compute::Cuda::Kernel>(_module->kernel("Cg3DPoissonSolve"));
				_supportsFullDeviceSolver = _supportsFullDeviceSolver && _fullDeviceSolver;
			}
			else
			{
				// Instantiate the right solver
#ifdef VCL_PHYSICS_FLUID_CUDA_SOLVER_CG
				_solver = std::make_unique<Vcl::Mathematics::Solver::ConjugateGradients>();
				_solver->setIterationChunkSize(10);
				_solver->setMaxIterations(10);
				_solver->setPrecision(0.0001);

				_solverCtx = std::make_unique<CenterGrid3DPoissonCgCtx>(_ownerCtx, _queue, Eigen::Vector3i(16, 16, 16));
#endif
#ifdef VCL_PHYSICS_FLUID_CUDA_SOLVER_JACOBI
				_solver = std::make_unique<Vcl::Mathematics::Solver::Jacobi>();
				_solver->setIterationChunkSize(10);
				_solver->setMaxIterations(10);
				_solver->setPrecision(0.0001);

				_solverCtx = std::make_unique<CenterGrid3DPoissonJacobiCtx>(_ownerCtx, _queue, Eigen::Vector3i(16, 16, 16));
#endif

				// Load the related kernels
				_compDiv         = static_pointer_cast<Compute::Cuda::Kernel>(_module->kernel("ComputeDivergence"));
				_buildLhs        = static_pointer_cast<Compute::Cuda::Kernel>(_module->kernel("BuildLHS"));
				_correctField    = static_pointer_cast<Compute::Cuda::Kernel>(_module->kernel("CorrectVelocities"));
				
				_supportsPartialDeviceSolver = _compDiv && _buildLhs && _correctField;
			}

			// Initialize the buffers
			unsigned int cells = 16 * 16 * 16;
			unsigned int mem_size = cells * sizeof(float);
			_laplacian[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[0]);
			_laplacian[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[1]);
			_laplacian[2] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[2]);
			_laplacian[3] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[3]);
			_pressure     = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_pressure);
			_divergence   = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_divergence);
		}

		Ensure(_supportsFullDeviceSolver || _supportsPartialDeviceSolver, "The solver is loaded.");
	}

	CenterGrid3DPoissonSolver::~CenterGrid3DPoissonSolver()
	{
		_ownerCtx->release(_module);
		_ownerCtx->release(_laplacian[0]);
		_ownerCtx->release(_laplacian[1]);
		_ownerCtx->release(_laplacian[2]);
		_ownerCtx->release(_laplacian[3]);
		_ownerCtx->release(_pressure);
		_ownerCtx->release(_divergence);
	}

	void CenterGrid3DPoissonSolver::solve(Fluid::CenterGrid& g)
	{
		Require(dynamic_cast<Cuda::CenterGrid*>(&g), "Grid is a CUDA grid.");

		Cuda::CenterGrid* grid = dynamic_cast<Cuda::CenterGrid*>(&g);
		if (!grid)
			return;

		int x = grid->resolution().x();
		int y = grid->resolution().y();
		int z = grid->resolution().z();
		int cells = x * y * z;
		Check(cells >= 0, "Dimensions are positive.");
		size_t mem_size = cells * sizeof(float);

		// Resize the internal solver buffers
		if (_laplacian[0]->size() < mem_size)
		{
			_ownerCtx->release(_laplacian[0]);
			_ownerCtx->release(_laplacian[1]);
			_ownerCtx->release(_laplacian[2]);
			_ownerCtx->release(_laplacian[3]);
			_ownerCtx->release(_pressure);
			_ownerCtx->release(_divergence);
		}

		// Fetch the internal solver buffers
		auto& laplacian0 = *_laplacian[0];
		auto& laplacian1 = *_laplacian[1];
		auto& laplacian2 = *_laplacian[2];
		auto& laplacian3 = *_laplacian[3];
		auto& pressure   = *_pressure;
		auto& divergence = *_divergence;

		_laplacian[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[0]);
		_laplacian[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[1]);
		_laplacian[2] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[2]);
		_laplacian[3] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[3]);
		_pressure     = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_pressure);
		_divergence   = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_divergence);


		// Fetch the grid data
		const auto& obstacles = grid->obstacleBuffer();

		auto& vel_x = grid->velocities(0, 0);
		auto& vel_y = grid->velocities(0, 1);
		auto& vel_z = grid->velocities(0, 2);

		// Execute the solver to ensure a divergence free velocity field
		if (_supportsFullDeviceSolver)
		{
			_fullDeviceSolver->run(*_queue, dim3(1, 1, 1), dim3(1, 1, 1), 0, (CUdeviceptr) pressure, dim3(x, y, z));
		}
		else if (_supportsPartialDeviceSolver)
		{
			dim3 block_size(16, 8, 1);
			dim3 grid_size(x / block_size.x, y / block_size.y, z / block_size.z);

			dim3 dimension(x, y, z);
			float cellSize = grid->spacing();
			float invCellSize = 1.0f / cellSize;

			std::array<ref_ptr<Compute::Buffer>, 4> laplacian =
			{
				_laplacian[0],
				_laplacian[1],
				_laplacian[2],
				_laplacian[3]
			};
			_solverCtx->setData
			(
				laplacian,
				grid->obstacles(),
				_pressure,
				_divergence,
				grid->resolution()
			);

			// Compute the divergence of the field
			_compDiv->run
			(
				*_queue,
				grid_size,
				block_size,
				0,
				(CUdeviceptr) divergence,
				(CUdeviceptr) pressure,
				(CUdeviceptr) vel_x,
				(CUdeviceptr) vel_y,
				(CUdeviceptr) vel_z,
				(CUdeviceptr) obstacles,
				dimension,
				cellSize
			);

			// Build the left hand side of the laplace equation
			_buildLhs->run
			(
				*_queue,
				grid_size,
				block_size,
				0,
				(CUdeviceptr) laplacian0,
				(CUdeviceptr) laplacian1,
				(CUdeviceptr) laplacian2,
				(CUdeviceptr) laplacian3,
				(CUdeviceptr) obstacles,
				0,
				false,
				dimension
			);

			// Compute the pressure field
			_solver->solve(_solverCtx.get());

			// Correct the velocity field with the computed pressure
			_correctField->run
			(
				*_queue,
				grid_size,
				block_size,
				0,
				(CUdeviceptr) vel_x,
				(CUdeviceptr) vel_y,
				(CUdeviceptr) vel_z,
				(CUdeviceptr) pressure,
				(CUdeviceptr) obstacles,
				dimension,
				invCellSize
			);
		}
	}
}}}}
#endif // VCL_CUDA_SUPPORT
