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
			_updateDensity = static_pointer_cast<Compute::Cuda::Kernel>(_fluidModule->kernel("SimpleDensityUpdate"));
		}
	}
	EulerFluidSimulation::~EulerFluidSimulation()
	{

	}

	void EulerFluidSimulation::update(Fluid::CenterGrid& g, float dt)
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

		// Clear the force buffers
		queue->setZero(force_x);
		queue->setZero(force_y);
		queue->setZero(force_z);

		// Wipe boundaries
		grid->setBorderZero(*queue, *vel_curr_x, res);
		grid->setBorderZero(*queue, *vel_curr_y, res);
		grid->setBorderZero(*queue, *vel_curr_z, res);

		// Update the density field
		// Move to update fields block
		grid->setBorderZero(*queue, *density_curr, res);
		grid->setBorderZero(*queue, *heat_curr, res);
		{
			const dim3 block_size(16, 4, 4);
			dim3 grid_size(res_x / 16, res_y / 4, res_z / 4);

			if (grid->heatDiffusion() > 0.0f)
			{
				_updateDensity->run
				(
					*queue,
					grid_size,
					block_size,
					0,
					obstacles,
					heat_curr,
					res
				);
			}
			else
			{
				_updateDensity->run
				(
					*queue,
					grid_size,
					block_size,
					0,
					obstacles,
					density_curr,
					res
				);
			}
		}

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
				vel_curr_x,
				vel_curr_y,
				vel_curr_z,
				obstacles,
				vort_x,
				vort_y,
				vort_z,
				vort,
				invSpacing,
				res
			);

			_addVorticity->run
			(
				*queue,
				grid_size,
				block_size,
				0,
				vort_x,
				vort_y,
				vort_z,
				vort,
				obstacles,
				force_x,
				force_y,
				force_z,
				grid->vorticityCoeff(),
				grid->spacing(),
				invSpacing,
				res
			);
		}

		// Add buoyancy
		// Move block to compute force method
		if (grid->heatDiffusion() > 0.0f)
		{
			grid->accumulate(*queue, *force_z, *heat_curr, grid->buoyancy());
		}
		else
		{
			grid->accumulate(*queue, *force_z, *density_curr, grid->buoyancy());
		}

		// Add forces
		grid->accumulate(*queue, *vel_curr_x, *force_x, dt);
		grid->accumulate(*queue, *vel_curr_y, *force_y, dt);
		grid->accumulate(*queue, *vel_curr_z, *force_z, dt);
		
		// Prepare the solver
		_poissonSolver->updateSolver(g);

		// Project into divergence free field
		_poissonSolver->makeDivergenceFree(g);

		// Can be combined with top accumulate?
		if (grid->heatDiffusion() > 0.0f)
		{
			grid->copyBorderX(*queue, *heat_prev, res);
			grid->copyBorderY(*queue, *heat_prev, res);
			grid->copyBorderZ(*queue, *heat_prev, res);

			const float heat_const = dt * grid->heatDiffusion() / (grid->spacing() * grid->spacing());

			// Diffuse the heat field
			_poissonSolver->diffuseField(g, heat_const);
		}

		// Advect
		grid->copyBorderX(*queue, *vel_curr_x, res);
		grid->copyBorderY(*queue, *vel_curr_y, res);
		grid->copyBorderZ(*queue, *vel_curr_z, res);
		
		// Configure the advection algorithm
		_advection->setSize(res_x, res_y, res_z);

		// Advect from current to prev
		const float dt0 = dt / grid->spacing();
		(*_advection)(queue, dt0, grid, vel_curr_x, vel_prev_x);
		(*_advection)(queue, dt0, grid, vel_curr_y, vel_prev_y);
		(*_advection)(queue, dt0, grid, vel_curr_z, vel_prev_z);

		// Move to external advection block
		(*_advection)(queue, dt0, grid, heat_curr, heat_prev);
		(*_advection)(queue, dt0, grid, density_curr, density_prev);

		// Fix border in updated fields
		grid->copyBorderX(*queue, *vel_prev_x, res);
		grid->copyBorderY(*queue, *vel_prev_y, res);
		grid->copyBorderZ(*queue, *vel_prev_z, res);
		grid->setBorderZero(*queue, *density_prev, res);
		grid->setBorderZero(*queue, *heat_prev, res);
		
		// Swap current and prev
		grid->swap();
	}
}}}}
#endif // VCL_CUDA_SUPPORT
