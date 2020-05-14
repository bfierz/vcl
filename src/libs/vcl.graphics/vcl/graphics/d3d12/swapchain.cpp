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
#include <vcl/graphics/d3d12/swapchain.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/d3d12/device.h>

namespace Vcl { namespace Graphics { namespace D3D12
{
	using namespace Microsoft::WRL;

	namespace
	{
		ComPtr<IDXGIFactory4> queryDXGIFactory()
		{
			ComPtr<IDXGIFactory4> factory;
			UINT flags = 0;
#ifdef VCL_DEBUG
			flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
			VCL_DIRECT3D_SAFE_CALL(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory)));
			return factory;
		}

		bool checkTearingSupport()
		{
			BOOL allowTearing = FALSE;

			// Rather than create the DXGI 1.5 factory interface directly, we create the
			// DXGI 1.4 interface and query for the 1.5 interface. This is to enable the
			// graphics debugging tools which will not support the 1.5 factory interface
			// until a future update.
			ComPtr<IDXGIFactory4> factory4;
			if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory4))))
			{
				ComPtr<IDXGIFactory5> factory5;
				if (SUCCEEDED(factory4.As(&factory5)))
				{
					if (FAILED(factory5->CheckFeatureSupport(
						DXGI_FEATURE_PRESENT_ALLOW_TEARING,
						&allowTearing, sizeof(allowTearing))))
					{
						allowTearing = FALSE;
					}
				}
			}

			return allowTearing == TRUE;
		}
	}

	SwapChain::SwapChain(Device* device, CommandQueue* queue, const SwapChainDescription& desc)
	: _device{ device }
	, _desc{ desc }
	, _syncPrimitive{ device->nativeDevice() }
	{
		ComPtr<IDXGIFactory4> factory = queryDXGIFactory();

		_VSync = desc.VSync;
		_tearing = checkTearingSupport();

		DXGI_SWAP_CHAIN_DESC1 d3d12_swapchain_desc = {};
		d3d12_swapchain_desc.Width = desc.Width;
		d3d12_swapchain_desc.Height = desc.Height;
		d3d12_swapchain_desc.Format = desc.ColourFormat;
		d3d12_swapchain_desc.Stereo = FALSE;
		d3d12_swapchain_desc.SampleDesc = { 1, 0 };
		d3d12_swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		d3d12_swapchain_desc.BufferCount = desc.NumberOfImages;
		d3d12_swapchain_desc.Scaling = DXGI_SCALING_STRETCH;
		d3d12_swapchain_desc.SwapEffect = desc.PresentMode;
		d3d12_swapchain_desc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
		d3d12_swapchain_desc.Flags = (_tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0) | DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

		ComPtr<IDXGISwapChain1> d3d12_swap_chain;
		VCL_DIRECT3D_SAFE_CALL(factory->CreateSwapChainForHwnd(
			queue->nativeQueue(),
			desc.Surface,
			&d3d12_swapchain_desc,
			nullptr,
			nullptr,
			&d3d12_swap_chain
		));
		VCL_DIRECT3D_SAFE_CALL(d3d12_swap_chain.As(&_d3d12SwapChain));
		VCL_DIRECT3D_SAFE_CALL(factory->MakeWindowAssociation(desc.Surface, DXGI_MWA_NO_ALT_ENTER));

		_fenceValues.resize(desc.NumberOfImages);
		_d3d12SwapChain->SetMaximumFrameLatency(desc.NumberOfImages);
		_currentBackBuffer = _d3d12SwapChain->GetCurrentBackBufferIndex();
		_frameLatencyWaitObject = _d3d12SwapChain->GetFrameLatencyWaitableObject();

		createRenderTargetViews(device, d3d12_swapchain_desc);
	}

	uint32_t SwapChain::waitForNextFrame()
	{
		// Wait for a free rendertarget
		_currentBackBuffer = _d3d12SwapChain->GetCurrentBackBufferIndex();
		if (_fenceValues[_currentBackBuffer] != 0)
		{
			_syncPrimitive.wait(_fenceValues[_currentBackBuffer], _frameLatencyWaitObject);
		}

		return _currentBackBuffer;
	}

	D3D12_CPU_DESCRIPTOR_HANDLE SwapChain::prepareFrame(ID3D12GraphicsCommandList* cmd_list)
	{
		// Render-targets are either being presented or used for rendering.
		// Transition here from presentation to rendering to use it for rendering.
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
			_backBuffers[_currentBackBuffer].Get(),
			D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
		cmd_list->ResourceBarrier(1, &barrier);

		// Create a handle to the current frame's render-target
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(
			_d3d12DescriptorHeap->GetCPUDescriptorHandleForHeapStart(), _currentBackBuffer, _descriptorIncrement);

		return rtv;
	}

	void SwapChain::present(CommandQueue* queue, ID3D12GraphicsCommandList* cmd_list, bool blocking)
	{
		// Render-targets are either being presented or used for rendering.
		// Transition here from rendering to presenting.
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
			_backBuffers[_currentBackBuffer].Get(),
			D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
		cmd_list->ResourceBarrier(1, &barrier);
		VCL_DIRECT3D_SAFE_CALL(cmd_list->Close());

		ID3D12CommandList* const generic_lists[] = { cmd_list };
		queue->nativeQueue()->ExecuteCommandLists(1, generic_lists);

		UINT sync_interval = _VSync ? 1 : 0;
		UINT present_flags = _tearing && !_VSync ? DXGI_PRESENT_ALLOW_TEARING : 0;
		VCL_DIRECT3D_SAFE_CALL(_d3d12SwapChain->Present(sync_interval, present_flags));

		_fenceValues[_currentBackBuffer] = _syncPrimitive.signal(queue->nativeQueue());
		if (blocking)
		{
			_syncPrimitive.wait(_fenceValues[_currentBackBuffer], _frameLatencyWaitObject);
		}
	}

	void SwapChain::resize(CommandQueue* queue, uint32_t width, uint32_t height)
	{
		releaseRenderTargetViews();

		DXGI_SWAP_CHAIN_DESC1 desc;
		_d3d12SwapChain->GetDesc1(&desc);
		_desc.Width = desc.Width = width;
		_desc.Height = desc.Height = height;

		ComPtr<IDXGIFactory4> factory;
		_d3d12SwapChain->GetParent(IID_PPV_ARGS(&factory));
		_d3d12SwapChain.Reset();
		CloseHandle(_frameLatencyWaitObject);

		ComPtr<IDXGISwapChain1> d3d12_swap_chain;
		VCL_DIRECT3D_SAFE_CALL(factory->CreateSwapChainForHwnd(
			queue->nativeQueue(),
			_desc.Surface,
			&desc,
			nullptr,
			nullptr,
			&d3d12_swap_chain
		));
		VCL_DIRECT3D_SAFE_CALL(d3d12_swap_chain.As(&_d3d12SwapChain));

		_desc.Width = width;
		_desc.Height = height;
		_d3d12SwapChain->SetMaximumFrameLatency(desc.BufferCount);
		_currentBackBuffer = _d3d12SwapChain->GetCurrentBackBufferIndex();
		_frameLatencyWaitObject = _d3d12SwapChain->GetFrameLatencyWaitableObject();

		createRenderTargetViews(_device, desc);
	}

	void SwapChain::wait()
	{
		if (_fenceValues[_currentBackBuffer] > 0)
			_syncPrimitive.wait(_fenceValues[_currentBackBuffer]);
	}

	void SwapChain::createRenderTargetViews(Device* device, const DXGI_SWAP_CHAIN_DESC1& desc)
	{
		auto d3d12_device = device->nativeDevice();
		_descriptorIncrement = d3d12_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

		_d3d12DescriptorHeap = _device->createDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, desc.BufferCount, false);
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handle(_d3d12DescriptorHeap->GetCPUDescriptorHandleForHeapStart());

		_backBuffers.reserve(desc.BufferCount);
		for (int i = 0; i < desc.BufferCount; ++i)
		{
			ComPtr<ID3D12Resource> bb;
			VCL_DIRECT3D_SAFE_CALL(_d3d12SwapChain->GetBuffer(i, IID_PPV_ARGS(&bb)));
			_backBuffers.emplace_back(bb);

			d3d12_device->CreateRenderTargetView(bb.Get(), nullptr, rtv_handle);
			rtv_handle.Offset(_descriptorIncrement);
		}
	}

	void SwapChain::releaseRenderTargetViews()
	{
		_d3d12DescriptorHeap.Reset();
		_backBuffers.clear();
	}
}}}
