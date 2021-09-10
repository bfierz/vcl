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
#include <vcl/config/direct3d12.h>

// C++ standard library
#include <chrono>
#include <vector>

// Windows Runtime Library
#include <wrl.h>

// Windows Graphics
#include <dxgi1_6.h>

// VCL
#include <vcl/graphics/d3d12/commandqueue.h>
#include <vcl/graphics/d3d12/device.h>
#include <vcl/graphics/d3d12/semaphore.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	struct SwapChainDescription
	{
		//! Handle to the surface used
		HWND Surface;

		//! Number of images
		uint32_t NumberOfImages;

		//! Select colour format
		DXGI_FORMAT ColourFormat;

		//! Requested width
		uint32_t Width;

		//! Requested height
		uint32_t Height;

		//! Mode to present image
		DXGI_SWAP_EFFECT PresentMode;

		//! Enable V-Sync
		bool VSync;
	};

	class SwapChain
	{
	public:
		template<typename T>
		using ComPtr = Microsoft::WRL::ComPtr<T>;

		SwapChain(Device* device, CommandQueue* queue, const SwapChainDescription& desc);

		std::pair<uint32_t, uint32_t> bufferSize() const { return std::make_pair(_desc.Width, _desc.Height); }

		uint32_t currentBufferIndex() const { return _currentBackBuffer; }
		ID3D12Resource* buffer(int idx) const { return _backBuffers[idx].Get(); }

		uint32_t waitForNextFrame();
		D3D12_CPU_DESCRIPTOR_HANDLE prepareFrame(ID3D12GraphicsCommandList* cmd_list);
		void present(CommandQueue* queue, ID3D12GraphicsCommandList* cmd_list, bool blocking);
		void resize(CommandQueue* queue, uint32_t width, uint32_t height);
		void wait();

	private:
		void createRenderTargetViews(Device* device, const DXGI_SWAP_CHAIN_DESC1& desc);
		void releaseRenderTargetViews();

		//! Associated device
		Device* _device;

		//! Description
		SwapChainDescription _desc;

		//! Back-buffers
		std::vector<ComPtr<ID3D12Resource>> _backBuffers;

		//! Current back-buffer
		uint32_t _currentBackBuffer;

		//! Descriptor heap increment
		uint32_t _descriptorIncrement;

		//! Enable VSync
		bool _VSync;

		//! Allow screen-tearing
		bool _tearing;

		//! Synchronization primitive
		Semaphore _syncPrimitive;

		//! Event to track free frame-buffers
		HANDLE _frameLatencyWaitObject;

		//! Fence values
		std::vector<uint64_t> _fenceValues;

		//! Native swap-chain object
		ComPtr<IDXGISwapChain4> _d3d12SwapChain;

		//! Descriptor heap for render target views
		ComPtr<ID3D12DescriptorHeap> _d3d12DescriptorHeap;
	};
}}}
