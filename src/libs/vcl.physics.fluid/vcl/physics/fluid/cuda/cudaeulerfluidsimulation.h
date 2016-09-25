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
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <array>

// VCL
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/module.h>
#include <vcl/physics/fluid/cuda/cudacentergrid.h>
#include <vcl/physics/fluid/cuda/cudamaccormackadvection.h>
#include <vcl/physics/fluid/centergrid.h>
#include <vcl/physics/fluid/eulerfluidsimulation.h>

namespace Vcl { namespace Physics { namespace Fluid { class CenterGrid3DPoissonSolver; }}}

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	/*!
	 *	\brief Simple grid based fluid simulation
	 */
	class EulerFluidSimulation : Fluid::EulerFluidSimulation
	{
	public:
		template<typename T> using ref_ptr = Vcl::Core::ref_ptr<T>;

	public:
		EulerFluidSimulation(ref_ptr<Vcl::Compute::Cuda::Context> ctx);
		virtual ~EulerFluidSimulation();

	public:
		virtual void update(Fluid::CenterGrid& g, float dt) override;

	private:

		//! Link to the context the solver was created with
		ref_ptr<Vcl::Compute::Cuda::Context> _ownerCtx = nullptr;

		//! Poisson grid solver
		std::unique_ptr<Fluid::CenterGrid3DPoissonSolver> _poissonSolver;

		//! Advection algorithm
		std::unique_ptr<Advection> _advection;

	private:

		//! Module with the cuda fluid code
		ref_ptr<Compute::Module> _fluidModule;

		//! Device function computing the vorticity of the velocity field
		ref_ptr<Vcl::Compute::Cuda::Kernel> _compVorticity = nullptr;

		//! Device function adding the vorticity to the force field
		ref_ptr<Vcl::Compute::Cuda::Kernel> _addVorticity = nullptr;
	};
}}}}
