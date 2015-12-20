/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <vcl/physics/fluid/cuda/cudaeulerfluidsimulation.h>

// C++ standard library
#include <iostream>

// VCL
#include <vcl/compute/cuda/module.h>
#include <vcl/core/contract.h>
#include <vcl/physics/fluid/cuda/cudacentergrid.h>
#include <vcl/physics/fluid/cuda/cudapoisson3dsolver.h>
#include <vcl/physics/fluid/openmp/omppoisson3dsolver.h>

#ifdef VCL_CUDA_SUPPORT
extern uint32_t EulerfluidSimulationCudaModule [];
extern size_t EulerfluidSimulationCudaModuleSize;


namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	EulerFluidSimulation::EulerFluidSimulation(ref_ptr<Vcl::Compute::Cuda::Context> ctx)
	: Fluid::EulerFluidSimulation()
	, _ownerCtx(ctx)
	{
		// Fetch the command queue
		auto queue = static_pointer_cast<Compute::Cuda::CommandQueue>(_ownerCtx->defaultQueue());

		_poissonSolver = std::make_unique<Cuda::CenterGrid3DPoissonSolver>(ctx, queue);
		//_poissonSolver = std::make_unique<OpenMP::CenterGrid3DPoissonSolver>();
		//_advection = std::make_unique<SemiLagrangeAdvection>(ctx);
		_advection = std::make_unique<MacCormackAdvection>(ctx);

		// Load the module
		_fluidModule = ctx->createModuleFromSource((int8_t*) EulerfluidSimulationCudaModule, EulerfluidSimulationCudaModuleSize*sizeof(uint32_t));

		if (_fluidModule)
		{
			// Load the fluid simulation related kernels
			_compVorticity = static_pointer_cast<Compute::Cuda::Kernel>(_fluidModule->kernel("ComputeVorticity"));
			_addVorticity  = static_pointer_cast<Compute::Cuda::Kernel>(_fluidModule->kernel("AddVorticity"));
		}
	}
	EulerFluidSimulation::~EulerFluidSimulation()
	{

	}

	void EulerFluidSimulation::update(Fluid::CenterGrid& g, double dt)
	{
		using namespace Eigen;
		using namespace Vcl::Compute;

		Require(dynamic_cast<Cuda::CenterGrid*>(&g), "Grid is a CUDA grid.");

		Cuda::CenterGrid* grid = dynamic_cast<Cuda::CenterGrid*>(&g);
		if (!grid)
			return;

		// Fetch the command queue
		auto queue = static_pointer_cast<Compute::Cuda::CommandQueue>(_ownerCtx->defaultQueue());

		// Buffers to run the simulation on
		auto force_x = grid->forces(0);
		auto force_y = grid->forces(1);
		auto force_z = grid->forces(2);

		auto vel_curr_x = grid->velocities(0, 0);
		auto vel_curr_y = grid->velocities(0, 1);
		auto vel_curr_z = grid->velocities(0, 2);
		auto vel_prev_x = grid->velocities(1, 0);
		auto vel_prev_y = grid->velocities(1, 1);
		auto vel_prev_z = grid->velocities(1, 2);

		auto obstacles = grid->obstacles();
		auto density_curr = grid->densities(0);
		auto density_prev = grid->densities(1);
		auto heat_curr = grid->heat(0);
		auto heat_prev = grid->heat(1);

		// Fetch the indexing data
		auto& res = grid->resolution();
		int res_x = res.x();
		int res_y = res.y();
		int res_z = res.z();
		int cells = res_x * res_y * res_z;
		Check(cells >= 0, "Dimensions are positive.");

		// Clear the force buffers
		queue->setZero(force_x);
		queue->setZero(force_y);
		queue->setZero(force_z);

		// Wipe boundaries
		grid->setBorderZero(*queue, *vel_curr_x, res);
		grid->setBorderZero(*queue, *vel_curr_y, res);
		grid->setBorderZero(*queue, *vel_curr_z, res);
		grid->setBorderZero(*queue, *density_curr, res);

		// Map the CUDA buffers for a prototype implementation
		/*auto* obstacle_ptr     = static_cast<unsigned int*>(obstacles.map());
		auto* density_curr_ptr = static_cast<float*>(density_curr.map());

		// Add an inlet
		for (size_t z = 1; z < 8; z++)
		{
			for (size_t y = res_y / 2 - 5; y < res_y / 2 + 5; y++)
			{
				for (size_t x = res_x / 2 - 5; x < res_x / 2 + 5; x++)
				{
					size_t index = z*res_x*res_y + y*res_x + x;
					float d = density_curr_ptr[index];
					density_curr_ptr[index] = fmax(d + 0.5f, 1.0f);
				}
			}
		}

		// Clean-up density field
		size_t index = res_x*res_y + res_x + 1;
		for (size_t z = 1; z < res_z - 1; z++, index += 2 * res_x)
		{
			for (size_t y = 1; y < res_y - 1; y++, index += 2)
			{
				for (size_t x = 1; x < res_x - 1; x++, index++)
				{
					if (obstacle_ptr[index])
					{
						density_curr_ptr[index] = 0.0f;
					}
				}
			}
		}

		density_curr.unmap();
		obstacles.unmap();*/

		CUevent start, stop;
		VCL_CU_SAFE_CALL(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
		VCL_CU_SAFE_CALL(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

		VCL_CU_SAFE_CALL(cuEventRecord(start, 0));

		// Add vorticity 
		if (grid->vorticityCoeff() > 0.0f)
		{
			auto& vort   = grid->vorticityMag();
			auto& vort_x = grid->vorticity(1);
			auto& vort_y = grid->vorticity(2);
			auto& vort_z = grid->vorticity(2);

			float invSpacing = 0.5f / grid->spacing();

			const dim3 block_size(16, 4, 4);
			dim3 grid_size(res_x / 16, res_y / 4, res_z / 4);

			_compVorticity->run
			(
				*queue,
				grid_size,
				block_size,
				0,
				(CUdeviceptr) *vel_curr_x,
				(CUdeviceptr) *vel_curr_y,
				(CUdeviceptr) *vel_curr_z,
				(CUdeviceptr) *obstacles,
				(CUdeviceptr) vort_x,
				(CUdeviceptr) vort_y,
				(CUdeviceptr) vort_z,
				(CUdeviceptr) vort,
				invSpacing,
				res
			);

			_addVorticity->run
			(
				*queue,
				grid_size,
				block_size,
				0,
				(CUdeviceptr) vort_x,
				(CUdeviceptr) vort_y,
				(CUdeviceptr) vort_z,
				(CUdeviceptr) vort,
				(CUdeviceptr) obstacles,
				(CUdeviceptr) force_x,
				(CUdeviceptr) force_y,
				(CUdeviceptr) force_z,
				grid->vorticityCoeff(),
				grid->spacing(),
				invSpacing,
				res
			);
		}

		// Add buoyancy
		if (grid->heatDiffusion() > 0.0f)
		{
			//	addBuoyancy(mHeat);
		}
		else
		{
			grid->accumulate(*queue, *force_z, *density_curr, grid->buoyancy());
		}

		// Add forces
		grid->accumulate(*queue, *vel_curr_x, *force_x, dt);
		grid->accumulate(*queue, *vel_curr_y, *force_y, dt);
		grid->accumulate(*queue, *vel_curr_z, *force_z, dt);

		// Project into divergence free field
		_poissonSolver->solve(g);

		if (grid->heatDiffusion() > 0.0f)
		{
		//	diffuseHeat(dt);
		}

		// Advect
		grid->copyBorderX(*queue, *vel_curr_x, res);
		grid->copyBorderY(*queue, *vel_curr_y, res);
		grid->copyBorderZ(*queue, *vel_curr_z, res);
		
		// Configure the advection algorithm
		_advection->setSize(res_x, res_y, res_z);

		const float dt0 = dt / grid->spacing();
		(*_advection)(queue, dt0, grid, heat_curr,    heat_prev);
		(*_advection)(queue, dt0, grid, density_curr, density_prev);
		(*_advection)(queue, dt0, grid, vel_curr_x,   vel_prev_x);
		(*_advection)(queue, dt0, grid, vel_curr_y,   vel_prev_y);
		(*_advection)(queue, dt0, grid, vel_curr_z,   vel_prev_z);

		grid->copyBorderX(*queue, *vel_prev_x, res);
		grid->copyBorderY(*queue, *vel_prev_y, res);
		grid->copyBorderZ(*queue, *vel_prev_z, res);
		grid->setBorderZero(*queue, *density_prev, res);
		grid->setBorderZero(*queue, *heat_prev, res);

		grid->swap();

		VCL_CU_SAFE_CALL(cuEventRecord(stop, 0));

		float time;
		VCL_CU_SAFE_CALL(cuEventSynchronize(stop));
		VCL_CU_SAFE_CALL(cuEventElapsedTime(&time, start, stop));
		std::cout << time << std::endl;
	}
}}}}
#endif // VCL_CUDA_SUPPORT
