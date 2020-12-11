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
#include <vcl/graphics/webgpu/swapchain.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/webgpu/webgpu.h>

namespace Vcl { namespace Graphics { namespace WebGPU
{
	SwapChain::SwapChain(WGPUDevice device, const SwapChainDescription& desc)
	: _device{ device }
	, _desc{ desc }
	{
		WGPUFenceDescriptor fence_desc = { nullptr, nullptr, 0 };
		_syncPrimitive = wgpuQueueCreateFence(wgpuDeviceGetDefaultQueue(_device), &fence_desc);

		resize(wgpuDeviceGetDefaultQueue(_device), desc.Width, desc.Height);
	}

	SwapChain::~SwapChain()
	{
		wgpuSwapChainRelease(_swapChain);
	}

	void SwapChain::present(WGPUQueue queue, bool blocking)
	{
#ifndef VCL_ARCH_WEBASM
		wgpuSwapChainPresent(_swapChain);
#endif
		wgpuQueueSignal(queue, _syncPrimitive, ++_frameCounter);
	}

	void SwapChain::resize(WGPUQueue queue, uint32_t width, uint32_t height)
	{
		if (_swapChain)
			wgpuSwapChainRelease(_swapChain);

#ifdef VCL_ARCH_WEBASM
		WGPUSwapChainDescriptor wgpu_desc = {};
		wgpu_desc.usage = WGPUTextureUsage_OutputAttachment;
		wgpu_desc.format = WGPUTextureFormat_RGBA8Unorm;
		wgpu_desc.width = width;
		wgpu_desc.height = height;
		wgpu_desc.presentMode = (WGPUPresentMode) _desc.PresentMode;

		_swapChain = wgpuDeviceCreateSwapChain(_device, _desc.Surface, &wgpu_desc);
#else
		_swapChainImpl = dawn_native::d3d12::CreateNativeSwapChainImpl(_device, reinterpret_cast<HWND>(_desc.NativeSurfaceHandle));
		WGPUSwapChainDescriptor wgpu_desc = {};
		wgpu_desc.implementation = reinterpret_cast<uint64_t>(&_swapChainImpl);
		_swapChain = wgpuDeviceCreateSwapChain(_device, nullptr, &wgpu_desc);

		wgpuSwapChainConfigure
		(
			_swapChain,
			dawn_native::d3d12::GetNativeSwapChainPreferredFormat(&_swapChainImpl),
			WGPUTextureUsage_OutputAttachment,
			width, height
		);
#endif
	}

	void SwapChain::wait()
	{
#ifndef VCL_ARCH_WEBASM
		// Wait for the last pending presentation
		while (wgpuFenceGetCompletedValue(_syncPrimitive) < _frameCounter)
			wgpuDeviceTick(_device);
#endif
	}
}}}
