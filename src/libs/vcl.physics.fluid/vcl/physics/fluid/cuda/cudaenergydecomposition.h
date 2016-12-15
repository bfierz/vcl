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
#pragma once

// VCL configuration
#include <vcl/config/eigen.h>

// C++ standard library
#include <array>

// VCL
#include <vcl/compute/cuda/module.h>
#include <vcl/physics/fluid/centergrid.h>

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	class EnergyDecomposition
	{
	public:
		EnergyDecomposition(ref_ptr<Vcl::Compute::Cuda::Context> ctx);
		virtual ~EnergyDecomposition();

	public:
		virtual void compute(Fluid::CenterGrid& g);

	private:
		void computeGold(const float* velx, const float* vely, const float* velz, unsigned int* obstacles, float* solution, const Eigen::Vector3i& dim);

	private:
		ref_ptr<Vcl::Compute::Cuda::Context> _ownerCtx;

	private: // Kernel functions

		//! Module with the cuda energy decomposition code
		ref_ptr<Vcl::Compute::Module> _energyModule;

		//! Device function computing the energy density of the velocity field
		ref_ptr<Vcl::Compute::Cuda::Kernel> _computeEnergyDensity;


		ref_ptr<Vcl::Compute::Cuda::Kernel> _marchObstaclesModifyEnergy;
		ref_ptr<Vcl::Compute::Cuda::Kernel> _obstaclesUpdateObstacles;

		//! Device function removing the compute low frequency parts
		ref_ptr<Vcl::Compute::Cuda::Kernel> _decompose;
		ref_ptr<Vcl::Compute::Cuda::Kernel> _downsampleX;
		ref_ptr<Vcl::Compute::Cuda::Kernel> _downsampleY;
		ref_ptr<Vcl::Compute::Cuda::Kernel> _downsampleZ;
		ref_ptr<Vcl::Compute::Cuda::Kernel> _upsampleX;
		ref_ptr<Vcl::Compute::Cuda::Kernel> _upsampleY;
		ref_ptr<Vcl::Compute::Cuda::Kernel> _upsampleZ;
	};
}}}}
