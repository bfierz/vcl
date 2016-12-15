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
#include <vcl/physics/fluid/cuda/cudawaveletturbulencefluidsimulation.h>

// C++ Standard library
#include <cstring>

// VCL
#include <vcl/core/contract.h>
#include <vcl/physics/fluid/cuda/cudacentergrid.h>
#include <vcl/util/vectornoise.h>

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	WaveletTrubulenceFluidSimulation::WaveletTrubulenceFluidSimulation(ref_ptr<Vcl::Compute::Cuda::Context> ctx)
	: _ownerCtx(ctx)
	{
		// Initialise the noise data on the CUDA device
		Vcl::Util::VectorNoise<32> noise;
		initNoiseBuffer(noise);
	}

	WaveletTrubulenceFluidSimulation::~WaveletTrubulenceFluidSimulation()
	{
	}

	void WaveletTrubulenceFluidSimulation::initNoiseBuffer(const Vcl::Util::VectorNoise<32>& noise)
	{
		int size = noise.size();
		std::array<const float*, 3> noise_tiles;
		noise.noiseData(&noise_tiles[0], &noise_tiles[1], &noise_tiles[2]);

		// Copy the noise to the device buffer
		for (int i = 0; i < 3; i++)
		{
			_noiseChannels[i] = static_pointer_cast<Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Compute::BufferAccess::Write, size*size*size*sizeof(float)));
			_ownerCtx->defaultQueue()->write(_noiseChannels[i], (void*) noise_tiles[i]);
		}
	}
}}}}
