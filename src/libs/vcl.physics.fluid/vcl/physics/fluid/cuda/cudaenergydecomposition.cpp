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
#include <vcl/physics/fluid/cuda/cudaenergydecomposition.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/math/waveletstack3d.h>
#include <vcl/physics/fluid/cuda/cudacentergrid.h>

#ifdef VCL_CUDA_SUPPORT
extern uint32_t EulerfluidEnergyDecompositionCudaModule [];
extern size_t EulerfluidEnergyDecompositionCudaModuleSize;

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	EnergyDecomposition::EnergyDecomposition(ref_ptr<Vcl::Compute::Cuda::Context> ctx)
	: _ownerCtx(ctx)
	{
		// Load the module
		_energyModule = ctx->createModuleFromSource((int8_t*)EulerfluidEnergyDecompositionCudaModule, EulerfluidEnergyDecompositionCudaModuleSize*sizeof(uint32_t));

		if (_energyModule)
		{
			// Load the fluid simulation related kernels
			_computeEnergyDensity       = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_engery_compute_energy_density"));
			_marchObstaclesModifyEnergy = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_energy_march_obstacles_modify_energy"));
			_obstaclesUpdateObstacles   = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_energy_march_obstacles_update_obstacle"));

			_decompose   = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_energy_decompose"));
			_downsampleX = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_energy_downsample_x"));
			_downsampleY = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_energy_downsample_y"));
			_downsampleZ = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_energy_downsample_z"));
			_upsampleX   = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_energy_upsample_x"));
			_upsampleY   = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_energy_upsample_y"));
			_upsampleZ   = static_pointer_cast<Compute::Cuda::Kernel>(_energyModule->kernel("cuda_energy_upsample_z"));
		}
	}

	EnergyDecomposition::~EnergyDecomposition()
	{

	}

	void EnergyDecomposition::compute(Fluid::CenterGrid& g)
	{
		VclRequire(dynamic_cast<Cuda::CenterGrid*>(&g), "Grid is a CUDA grid.");
		VclRequire(g.resolution().x() <= 128 && g.resolution().y() <= 128 && g.resolution().z() <= 128, "Grid resolution is smaller than 128 per dimension.")

		Cuda::CenterGrid* grid = dynamic_cast<Cuda::CenterGrid*>(&g);
		if (!grid)
			return;

		// Fetch the command queue
		auto& queue = *static_pointer_cast<Compute::Cuda::CommandQueue>(_ownerCtx->defaultQueue());

		// Fetch the indexing data
		auto& res = grid->resolution();
		int res_x = res.x();
		int res_y = res.y();
		int res_z = res.z();
		int cells = res_x * res_y * res_z;
		VclCheck(cells >= 0, "Dimensions are positive.");

		// Fetch necessary buffers
		auto vel_x = grid->velocities(0, 0);
		auto vel_y = grid->velocities(0, 1);
		auto vel_z = grid->velocities(0, 2);
		auto obstacles = grid->obstacles();

		if (!grid->hasNamedBuffer("Energy"))
		{
			grid->addNamedBuffer("Energy");
		}
		auto energy = static_pointer_cast<Compute::Cuda::Buffer>(grid->namedBuffer("Energy"));

		{
			const dim3 block_size(16, 4, 4);
			dim3 grid_size(res_x / 16, res_y / 4, res_z / 4);

			_computeEnergyDensity->run
			(
				queue,
				grid_size,
				block_size,
				0,
				vel_x,
				vel_y,
				vel_z,
				energy,
				dim3(res_x, res_y, res_z)
			);
	
			// Pseudo-march the values into the obstacles.
			// The wavelet upsampler only uses a 3x3 support neighborhood, so
			// propagating the values in by 4 should be sufficient
			for (int iter = 0; iter < 4; iter++)
			{
				_marchObstaclesModifyEnergy->run
				(
					queue,
					grid_size,
					block_size,
					0,
					obstacles,
					energy,
					dim3(res_x, res_y, res_z)
				);
		
				_obstaclesUpdateObstacles->run
				(
					queue,
					grid_size,
					block_size,
					0,
					obstacles,
					dim3(res_x, res_y, res_z)
				);
			}
		}
	
		//
		//	Decompose the energy field
		//
		
		// Get two temporary buffers
		auto tmp1 = static_pointer_cast<Compute::Cuda::Buffer>(grid->aquireIntermediateBuffer());
		auto tmp2 = static_pointer_cast<Compute::Cuda::Buffer>(grid->aquireIntermediateBuffer());
		queue.setZero(tmp1);
		queue.setZero(tmp2);
		
		// Downsample the volume
		int max_threads = 128;
		{
			// Downsample X
			dim3 block_size(res_x / 2, max_threads / (res_x / 2), 1);
			dim3 grid_size(res_y / block_size.y, res_z);
		
			_downsampleX->run
			(
				queue,
				grid_size,
				block_size,
				0,
				tmp1,
				energy,
				dim3(res_x, res_y, res_z)
			);
		}
		{
			// Downsample Y
			dim3 block_size(res_x, max_threads / res_x, 1);
			dim3 grid_size(res_y / (2 * block_size.y), res_z);
		
			_downsampleY->run(queue, grid_size, block_size, 0, tmp2, tmp1, dim3(res_x, res_y, res_z));
		}
		{
			// Downsample Z
			dim3 block_size(res_x, max_threads / res_x, 1);
			dim3 grid_size(res_y / block_size.y, res_z / 2);
		
			_downsampleZ->run(queue, grid_size, block_size, 0, tmp1, tmp2, dim3(res_x, res_y, res_z));
		}
		
		{
			dim3 block_size(res_x, max_threads / res_x, 1);
			dim3 grid_size(res_y / block_size.y, res_z);
		
			// Upsample the volume
			_upsampleZ->run(queue, grid_size, block_size, 0, tmp2, tmp1, dim3(res_x, res_y, res_z));
			_upsampleY->run(queue, grid_size, block_size, 0, tmp1, tmp2, dim3(res_x, res_y, res_z));
			_upsampleX->run(queue, grid_size, block_size, 0, tmp2, tmp1, dim3(res_x, res_y, res_z));
		
			// Collect the results
			_decompose->run
			(
				queue,
				grid_size,
				block_size,
				0,
				tmp2,
				energy,
				dim3(res_x, res_y, res_z)
			);
		}

		grid->releaseIntermediateBuffer(tmp1);
		grid->releaseIntermediateBuffer(tmp2);

//#define VCL_PHYSICS_FLUID_CUDA_ENERGY_VERIFY
#ifdef VCL_PHYSICS_FLUID_CUDA_ENERGY_VERIFY
		//
		// Copy the Input buffers to the host. Used as input for the gold standard.
		//
		auto host_velocity_x = new float[res_x*res_y*res_z];
		auto host_velocity_y = new float[res_x*res_y*res_z];
		auto host_velocity_z = new float[res_x*res_y*res_z];
		auto host_obstacles  = new unsigned int[res_x*res_y*res_z];

		VCL_CU_SAFE_CALL(cuMemcpyDtoH(host_velocity_x, vel_x.devicePtr(), res_x*res_y*res_z*sizeof(float)));
		VCL_CU_SAFE_CALL(cuMemcpyDtoH(host_velocity_y, vel_y.devicePtr(), res_x*res_y*res_z*sizeof(float)));
		VCL_CU_SAFE_CALL(cuMemcpyDtoH(host_velocity_z, vel_z.devicePtr(), res_x*res_y*res_z*sizeof(float)));
		VCL_CU_SAFE_CALL(cuMemcpyDtoH(host_obstacles, obstacles.devicePtr(), res_x*res_y*res_z*sizeof(unsigned int)));

		// Compute the reference solution
		float* ref_energy = new float[res_x*res_y*res_z];
		computeGold(host_velocity_x, host_velocity_y, host_velocity_z, host_obstacles, ref_energy, Eigen::Vector3i(res_x, res_y, res_z));

		int mem_size = res_x*res_y*res_z*sizeof(float);
		float* cuda_energy = new float[res_x*res_y*res_z];
		VCL_CU_SAFE_CALL(cuMemcpyDtoH(cuda_energy, energy.devicePtr(), mem_size));

		float L1 = 0.0f;
		for (int i = 0; i < res_x*res_y*res_z; i++)
		{
			L1 += abs(ref_energy[i] - cuda_energy[i]);
		}

		std::cout << "Energy decomposition error: Total L1 = " << L1 << ", Per element: L1 = " << L1 / (float) (res_x*res_y*res_z) << std::endl;

		delete host_velocity_x;
		delete host_velocity_y;
		delete host_velocity_z;
		delete host_obstacles;
		delete ref_energy;
#endif // VCL_PHYSICS_FLUID_CUDA_ENERGY_VERIFY
	}


	void EnergyDecomposition::computeGold(const float* velx, const float* vely, const float* velz, unsigned int* obstacles, float* solution, const Eigen::Vector3i& dim)
	{
		enum OBSTACLE_FLAGS
		{
			EMPTY = 0, 
			MARCHED = 2, 
			RETIRED = 4 
		}; 

		size_t nr_cells = dim.x()*dim.y()*dim.z();

		// compute everywhere
		float* energy = new float[nr_cells];
		for (int i = 0; i < nr_cells; i++)
		{
			energy[i] = 0.5f * (velx[i] * velx[i] + vely[i] * vely[i] + velz[i] * velz[i]);
		}

		// pseudo-march the values into the obstacles
		// the wavelet upsampler only uses a 3x3 support neighborhood, so
		// propagating the values in by 4 should be sufficient
		for (int iter = 0; iter < 4; iter++)
		{
			int index = dim.x()*dim.y() + dim.x() + 1;
			for (int z = 1; z < dim.z() - 1; z++, index += 2 * dim.x())
			{
				for (int y = 1; y < dim.y() - 1; y++, index += 2)
				{
					for (int x = 1; x < dim.x() - 1; x++, index++)
					{
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
							if (!obstacles[index + dim.x()] || obstacles[index + dim.x()] == RETIRED)
							{
								sum += energy[index + dim.x()];
								valid++;
							}
							if (!obstacles[index - dim.x()] || obstacles[index - dim.x()] == RETIRED)
							{
								sum += energy[index - dim.x()];
								valid++;
							}
							if (!obstacles[index + dim.x()*dim.y()] || obstacles[index + dim.x()*dim.y()] == RETIRED)
							{
								sum += energy[index + dim.x()*dim.y()];
								valid++;
							}
							if (!obstacles[index - dim.x()*dim.y()] || obstacles[index - dim.x()*dim.y()] == RETIRED)
							{
								sum += energy[index - dim.x()*dim.y()];
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
			}

			index = dim.x()*dim.y() + dim.x() + 1;
			for (int z = 1; z < dim.z() - 1; z++, index += 2 * dim.x())
			{
				for (int y = 1; y < dim.y() - 1; y++, index += 2)
				{
					for (int x = 1; x < dim.x() - 1; x++, index++)
					{
						if (obstacles[index] == MARCHED)
							obstacles[index] = RETIRED;
					}
				}
			}
		}

		Vcl::Mathematics::WaveletStack3D wavelet_stack(dim.x(), dim.y(), dim.z());
		wavelet_stack.CreateSingleLayer(energy);

		float* wavelet_energy = wavelet_stack.Stack()[0];
		size_t mem_size = nr_cells*sizeof(float);

		memcpy(solution, wavelet_energy, mem_size);
	}
}}}}
#endif // VCL_CUDA_SUPPORT
