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
#include <vcl/compute/cuda/module.h>
#include <vcl/physics/fluid/centergrid.h>

// Forward declarations
namespace Vcl { namespace Util { template<int N> class VectorNoise; }}

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	class WaveletTrubulenceFluidSimulation
	{
	public:
		WaveletTrubulenceFluidSimulation(ref_ptr<Vcl::Compute::Cuda::Context> ctx);
		virtual ~WaveletTrubulenceFluidSimulation();

	private:
		void initNoiseBuffer(const Vcl::Util::VectorNoise<32>& noise);

	private:
		//! Link to the context the solver was created with
		ref_ptr<Vcl::Compute::Cuda::Context> _ownerCtx{ nullptr };

	private:
		//! Module with the cuda wavelet turbulence noise code
		ref_ptr<Vcl::Compute::Cuda::Module> _noiseModule;

	private:
		//! Random noise data
		std::array<ref_ptr<Vcl::Compute::Cuda::Buffer>, 3> _noiseChannels;
	};
}}}}
