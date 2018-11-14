#include <vcl/physics/fluid/cuda/cudapoisson3dsolver.h>

// VCL
#include <vcl/math/solver/cuda/poisson3dsolver_cg.h>
#include <vcl/math/solver/cuda/poisson3dsolver_jacobi.h>
#include <vcl/math/solver/conjugategradients.h>
#include <vcl/math/solver/jacobi.h>
#include <vcl/physics/fluid/cuda/cudacentergrid.h>
#include <vcl/physics/fluid/cuda/cudapoisson3dsolver_jacobi.h>

#ifdef VCL_CUDA_SUPPORT
extern uint32_t Poisson3dSolverCudaModule [];
extern size_t Poisson3dSolverCudaModuleSize;

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	CenterGrid3DPoissonSolver::CenterGrid3DPoissonSolver
	(
		ref_ptr<Vcl::Compute::Cuda::Context> ctx,
		ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue
	)
	: Fluid::CenterGrid3DPoissonSolver()
	, _ownerCtx(ctx)
	, _queue(queue)
	{
		VclRequire(ctx, "Context is set");

		// Is dynamic parallelism supported
		//_supportsFullDeviceSolver = ctx->supports(Vcl::Compute::Cuda::Feature::DynamicParallelism);
		
		// Load the module
		_module = static_pointer_cast<Compute::Cuda::Module>(ctx->createModuleFromSource((int8_t*) Poisson3dSolverCudaModule, Poisson3dSolverCudaModuleSize * sizeof(uint32_t)));

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

				_solverCtx = std::make_unique<Vcl::Mathematics::Solver::Cuda::Poisson3DCgCtx>(_ownerCtx, _queue, Eigen::Vector3ui(16, 16, 16));
#endif
#ifdef VCL_PHYSICS_FLUID_CUDA_SOLVER_JACOBI
				_solver = std::make_unique<Vcl::Mathematics::Solver::Jacobi>();
				_solver->setIterationChunkSize(10);
				_solver->setMaxIterations(10);
				_solver->setPrecision(0.0001);

				_solverCtx = std::make_unique<Vcl::Mathematics::Solver::Cuda::Poisson3DJacobiCtx>(_ownerCtx, _queue, Eigen::Vector3ui(16, 16, 16));
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

		VclEnsure(_supportsFullDeviceSolver || _supportsPartialDeviceSolver, "The solver is loaded.");
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

	void CenterGrid3DPoissonSolver::updateSolver(Fluid::CenterGrid& g)
	{
		VclRequire(dynamic_cast<Cuda::CenterGrid*>(&g), "Grid is a CUDA grid.");

		Cuda::CenterGrid* grid = dynamic_cast<Cuda::CenterGrid*>(&g);
		if (!grid)
			return;

		int x = grid->resolution().x();
		int y = grid->resolution().y();
		int z = grid->resolution().z();
		int cells = x * y * z;
		VclCheck(cells >= 0, "Dimensions are positive.");
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

			_laplacian[0] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[0]);
			_laplacian[1] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[1]);
			_laplacian[2] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[2]);
			_laplacian[3] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_laplacian[3]);
			_pressure = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_pressure);
			_divergence = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, mem_size)); _queue->setZero(_divergence);

#ifdef VCL_PHYSICS_FLUID_CUDA_SOLVER_CG
			_solverCtx = std::make_unique<Vcl::Mathematics::Solver::Cuda::Poisson3DCgCtx>(_ownerCtx, _queue, Eigen::Vector3ui(x, y, z));
#endif
#ifdef VCL_PHYSICS_FLUID_CUDA_SOLVER_JACOBI
			_solverCtx = std::make_unique<Vcl::Mathematics::Solver::Cuda::Poisson3DJacobiCtx>(_ownerCtx, _queue, Eigen::Vector3ui(x, y, z));
#endif
		}

		dim3 block_size(16, 8, 1);
		dim3 grid_size(x / block_size.x, y / block_size.y, z / block_size.z);
		
		// Problem domain
		dim3 dimension(x, y, z);

		// Fetch the internal solver buffers
		auto& laplacian0 = *_laplacian[0];
		auto& laplacian1 = *_laplacian[1];
		auto& laplacian2 = *_laplacian[2];
		auto& laplacian3 = *_laplacian[3];

		// Fetch the grid data
		const auto obstacles = grid->obstacles();

		// Build the left hand side of the laplace equation
		_buildLhs->run
		(
			*_queue,
			grid_size,
			block_size,
			0,
			laplacian0,
			laplacian1,
			laplacian2,
			laplacian3,
			obstacles,
			0,
			false,
			dimension
		);

		float cellSize = grid->spacing();
		_solverCtx->updatePoissonStencil(cellSize, -1, 0, *obstacles);
	}

	void CenterGrid3DPoissonSolver::makeDivergenceFree(Fluid::CenterGrid& g)
	{
		VclRequire(dynamic_cast<Cuda::CenterGrid*>(&g), "Grid is a CUDA grid.");

		Cuda::CenterGrid* grid = dynamic_cast<Cuda::CenterGrid*>(&g);
		if (!grid)
			return;

		int x = grid->resolution().x();
		int y = grid->resolution().y();
		int z = grid->resolution().z();
		VclCheck(x * y * z >= 0, "Dimensions are positive.");
		VclCheck(_laplacian[0]->size() <= x * y * z * sizeof(float), "Not enough memory allocated");

		// Fetch the internal solver buffers
		auto& pressure   = *_pressure;
		auto& divergence = *_divergence;
		
		// Fetch the grid data
		const auto obstacles = grid->obstacles();

		auto vel_x = grid->velocities(0, 0);
		auto vel_y = grid->velocities(0, 1);
		auto vel_z = grid->velocities(0, 2);

		// Execute the solver to ensure a divergence free velocity field
		if (_supportsFullDeviceSolver)
		{
			_fullDeviceSolver->run(*_queue, dim3(1, 1, 1), dim3(1, 1, 1), 0, pressure, dim3(x, y, z));
		}
		else if (_supportsPartialDeviceSolver)
		{
			dim3 block_size(16, 8, 1);
			dim3 grid_size(x / block_size.x, y / block_size.y, z / block_size.z);

			dim3 dimension(x, y, z);
			float cellSize = grid->spacing();
			float invCellSize = 1.0f / cellSize;

			// Compute the divergence of the field
			_compDiv->run
			(
				*_queue,
				grid_size,
				block_size,
				0,
				divergence,
				pressure,
				vel_x,
				vel_y,
				vel_z,
				obstacles,
				dimension,
				cellSize
			);

			// Compute the pressure field
			_solverCtx->setData
			(
				_pressure,
				_divergence
			);
			_solver->solve(_solverCtx.get());

			// Correct the velocity field with the computed pressure
			_correctField->run
			(
				*_queue,
				grid_size,
				block_size,
				0,
				vel_x,
				vel_y,
				vel_z,
				pressure,
				obstacles,
				dimension,
				invCellSize
			);
		}
	}

	void CenterGrid3DPoissonSolver::diffuseField(Fluid::CenterGrid& g, float diffusion_constant)
	{
		VclRequire(dynamic_cast<Cuda::CenterGrid*>(&g), "Grid is a CUDA grid.");

		Cuda::CenterGrid* grid = dynamic_cast<Cuda::CenterGrid*>(&g);
		if (!grid)
			return;

		////////////////////////////////////////////////////////////////////////
		int x = grid->resolution().x();
		int y = grid->resolution().y();
		int z = grid->resolution().z();
		dim3 block_size(16, 8, 1);
		dim3 grid_size(x / block_size.x, y / block_size.y, z / block_size.z);

		// Problem domain
		dim3 dimension(x, y, z);

		// Fetch the internal solver buffers
		auto& laplacian0 = *_laplacian[0];
		auto& laplacian1 = *_laplacian[1];
		auto& laplacian2 = *_laplacian[2];
		auto& laplacian3 = *_laplacian[3];

		// Fetch the grid data
		const auto obstacles = grid->obstacles();

		// Build the left hand side of the laplace equation
		_buildLhs->run
		(
			*_queue,
			grid_size,
			block_size,
			0,
			laplacian0,
			laplacian1,
			laplacian2,
			laplacian3,
			obstacles,
			diffusion_constant,
			true,
			dimension
		);

		float cellSize = grid->spacing();
		_solverCtx->updatePoissonStencil(cellSize, -diffusion_constant, 1, *obstacles);
		_solverCtx->setData
		(
			grid->heat(0),
			grid->heat(1)
		);

		_solver->solve(_solverCtx.get());
	}
}}}}
#endif // VCL_CUDA_SUPPORT
