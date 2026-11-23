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

namespace Vcl { namespace Graphics { namespace WebGPU {
	namespace {
		WGPUPresentMode toWebGPU(PresentMode mode)
		{
			switch (mode)
			{
			case PresentMode::Immediate:
				return WGPUPresentMode_Immediate;
			case PresentMode::Mailbox:
				return WGPUPresentMode_Mailbox;
			default:
				return WGPUPresentMode_Fifo;
			}
		}
	}

	SwapChain::SwapChain(WGPUDevice device, const SwapChainDescription& desc)
	: _device{ device }
	, _desc{ desc }
	{
		VclRequire(desc.Surface, "Surface is provided");

		_format = selectFormat();
		resize(desc.Width, desc.Height);
	}

	WGPUTextureFormat SwapChain::selectFormat() const
	{
		const auto requested = _desc.ColourFormat != SurfaceFormat::Unknown ? toWebGPUEnum(_desc.ColourFormat) : WGPUTextureFormat_Undefined;
		// The surface is queried on the adapter of the device it is used with
		WGPUAdapter adapter = wgpuDeviceGetAdapter(_device);
		WGPUSurfaceCapabilities caps = {};
		const auto status = wgpuSurfaceGetCapabilities(_desc.Surface, adapter, &caps);
		wgpuAdapterRelease(adapter);
		VclCheck(status == WGPUStatus_Success && caps.formatCount > 0, "Surface reports supported formats");
		if (status != WGPUStatus_Success || caps.formatCount == 0)
			return requested;

		WGPUTextureFormat format = caps.formats[0];
		for (size_t i = 0; i < caps.formatCount; i++)
		{
			if (caps.formats[i] == requested)
				format = requested;
		}
		wgpuSurfaceCapabilitiesFreeMembers(caps);
		return format;
	}

	SwapChain::~SwapChain()
	{
		releaseBackBuffer();
		wgpuSurfaceUnconfigure(_desc.Surface);
	}

	WGPUTextureView SwapChain::currentBackBuffer()
	{
		if (!_backBufferView)
		{
			WGPUSurfaceTexture surface_texture = {};
			wgpuSurfaceGetCurrentTexture(_desc.Surface, &surface_texture);
			if (surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
				surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
			{
				return nullptr;
			}

			_backBuffer = surface_texture.texture;
			_backBufferView = wgpuTextureCreateView(_backBuffer, nullptr);
		}
		return _backBufferView;
	}

	void SwapChain::present(WGPUQueue queue, bool blocking)
	{
#ifndef VCL_ARCH_WEBASM
		wgpuSurfacePresent(_desc.Surface);
#endif
		releaseBackBuffer();

		++_requestedFrame;

		WGPUQueueWorkDoneCallbackInfo callback = {};
		callback.mode = WGPUCallbackMode_AllowSpontaneous;
		callback.callback = swapChainWorkSubmittedCallback;
		callback.userdata1 = this;
		wgpuQueueOnSubmittedWorkDone(queue, callback);

		if (blocking)
			wait();
	}

	void SwapChain::resize(uint32_t width, uint32_t height)
	{
		releaseBackBuffer();

		_desc.Width = width;
		_desc.Height = height;

		WGPUSurfaceConfiguration config = {};
		config.device = _device;
		config.format = _format;
		config.usage = WGPUTextureUsage_RenderAttachment;
		config.width = width;
		config.height = height;
		config.alphaMode = WGPUCompositeAlphaMode_Auto;
		config.presentMode = _desc.VSync ? WGPUPresentMode_Fifo : toWebGPU(_desc.PresentMode);
		wgpuSurfaceConfigure(_desc.Surface, &config);
	}

	void SwapChain::wait()
	{
#ifndef VCL_ARCH_WEBASM
		// Wait for the last pending presentation
		while (_completedFrames < _requestedFrame)
			wgpuDeviceTick(_device);
#endif
	}

	void SwapChain::releaseBackBuffer()
	{
		if (_backBufferView)
			wgpuTextureViewRelease(_backBufferView);
		if (_backBuffer)
			wgpuTextureRelease(_backBuffer);
		_backBufferView = nullptr;
		_backBuffer = nullptr;
	}

	void SwapChain::swapChainWorkSubmittedCallback(WGPUQueueWorkDoneStatus, WGPUStringView, void* userdata1, void*)
	{
		auto swap_chain = static_cast<SwapChain*>(userdata1);
		swap_chain->_completedFrames = swap_chain->_requestedFrame;
	}
}}}
