/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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
#include <vcl/config/webgpu.h>

// C++ standard library
#include <chrono>
#include <vector>

// Abseil
#include <absl/container/inlined_vector.h>

#ifndef VCL_ARCH_WEBASM
// Dawn
#include <dawn_native/D3D12Backend.h>
#include <dawn_native/DawnNative.h>
#endif

// VCL
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace WebGPU
{
	enum class PresentMode
	{
		Immediate = 0x00000000,
		Mailbox = 0x00000001,
		Fifo = 0x00000002,
	};

	struct SwapChainDescription
	{
		//! Handle to the surface used
		WGPUSurface Surface;

		//! Native window handle
		uint64_t NativeSurfaceHandle;

		//! Number of images
		uint32_t NumberOfImages;

		//! Select colour format
		SurfaceFormat ColourFormat;

		//! Requested width
		uint32_t Width;

		//! Requested height
		uint32_t Height;

		//! Mode to present image
		PresentMode PresentMode;

		//! Enable V-Sync
		bool VSync;
	};

	class SwapChain
	{
	public:
		SwapChain(WGPUDevice device, const SwapChainDescription& desc);
		~SwapChain();

		std::pair<uint32_t, uint32_t> bufferSize() const { return std::make_pair(_desc.Width, _desc.Height); }
		WGPUTextureView currentBackBuffer() { return wgpuSwapChainGetCurrentTextureView(_swapChain); }

		void present(WGPUQueue queue, bool blocking);
		void resize(uint32_t width, uint32_t height);
		void wait();

	private:
		static void swapChainWorkSubmittedCallback(WGPUQueueWorkDoneStatus status, void* sc);
		
		//! Associated device
		WGPUDevice _device;

		//! Description
		SwapChainDescription _desc;

		//! Count frame completion requests
		uint64_t _requestedFrame{ 0 };

		//! Completed frames
		uint64_t _completedFrames{ 0 };

		//! Native swap-chain object
		WGPUSwapChain _swapChain{};

#ifndef VCL_ARCH_WEBASM
		//! Dawn swap-chain
		DawnSwapChainImplementation _swapChainImpl{};
#endif
	};
}}}
