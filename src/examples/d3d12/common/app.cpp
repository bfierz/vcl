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
#include "app.h"

// C++ standard library
#include <exception>
#include <stdexcept>

// VCL
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>

const int Application::NumberOfFrames = 3;

Application::Application(LPCSTR title)
{
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, _T("VCL D3D12 Example"), nullptr };
	::RegisterClassEx(&wc);
	_windowHandle = ::CreateWindow(wc.lpszClassName, title, WS_OVERLAPPEDWINDOW, 100, 100, 1280, 800, nullptr, nullptr, wc.hInstance, this);

	if (!initD3d12(_windowHandle))
	{
		::DestroyWindow(_windowHandle);
		::UnregisterClass(wc.lpszClassName, wc.hInstance);
		throw std::runtime_error("D3D12 failed to initialize");
	}
}

Application::~Application()
{
	_graphicsCommandBuffer.reset();

	_frames[0].GraphicsCommandAllocator.Reset();
	_frames[1].GraphicsCommandAllocator.Reset();
	_frames[2].GraphicsCommandAllocator.Reset();

	_swapChain.reset();
	_device.reset();

	::DestroyWindow(_windowHandle);
	::UnregisterClass(_T("VCL D3D12 Example"), GetModuleHandle(nullptr));
}

int Application::run()
{
	device()->defaultQueue()->sync();

	// Show the window
	::ShowWindow(_windowHandle, SW_SHOWDEFAULT);
	::UpdateWindow(_windowHandle);

	MSG msg;
	ZeroMemory(&msg, sizeof(msg));
	while (msg.message != WM_QUIT)
	{
		if (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
		{
			::TranslateMessage(&msg);
			::DispatchMessage(&msg);
			continue;
		}

		// Allow to update the state of objects before waiting for the GPU
		updateFrame();

		const auto frame_idx = _swapChain->waitForNextFrame();
		_frames[frame_idx].GraphicsCommandAllocator->Reset();
		_graphicsCommandBuffer->reset(_frames[frame_idx].GraphicsCommandAllocator.Get());
		auto rtv = _swapChain->prepareFrame(_graphicsCommandBuffer->handle());
		auto dsv = _dsvHeap->GetCPUDescriptorHandleForHeapStart();

		// Clear the back-buffer
		float clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
		_graphicsCommandBuffer->handle()->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
		_graphicsCommandBuffer->handle()->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

		renderFrame(_graphicsCommandBuffer.get(), rtv, dsv);
		renderFrame(_graphicsCommandBuffer->handle(), rtv, dsv);
		_swapChain->present(_device->defaultQueue(), _graphicsCommandBuffer->handle(), false);
	}

	_swapChain->wait();

	return EXIT_SUCCESS;
}

void Application::invalidateDeviceObjects()
{
	_depthBuffer.Reset();
	_dsvHeap.Reset();
}
void Application::createDeviceObjects()
{
	auto dev = _device->nativeDevice();

	D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
	dsvHeapDesc.NumDescriptors = 1;
	dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	VCL_DIRECT3D_SAFE_CALL(dev->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&_dsvHeap)));

	D3D12_CLEAR_VALUE optimizedClearValue = {};
	optimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
	optimizedClearValue.DepthStencil = { 1.0f, 0 };

	const auto size = _swapChain->bufferSize();
	VCL_DIRECT3D_SAFE_CALL(dev->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, size.first, size.second,
			1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
		D3D12_RESOURCE_STATE_DEPTH_WRITE,
		&optimizedClearValue,
		IID_PPV_ARGS(&_depthBuffer)
	));

	D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
	dsv.Format = DXGI_FORMAT_D32_FLOAT;
	dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	dsv.Texture2D.MipSlice = 0;
	dsv.Flags = D3D12_DSV_FLAG_NONE;

	dev->CreateDepthStencilView(_depthBuffer.Get(), &dsv,
		_dsvHeap->GetCPUDescriptorHandleForHeapStart());
}

bool Application::initD3d12(HWND hWnd)
{
	using namespace Vcl::Graphics::D3D12;

	SwapChainDescription sc_desc;
	sc_desc.Surface = hWnd;
	sc_desc.NumberOfImages = NumberOfFrames;
	sc_desc.ColourFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	sc_desc.Width = 0;
	sc_desc.Height = 0;
	sc_desc.PresentMode = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	sc_desc.VSync = true;

	_device = std::make_unique<Device>(DeviceType::Hardware);
	_swapChain = std::make_unique<SwapChain>(_device.get(), _device->defaultQueue(), sc_desc);

	_frames[0].GraphicsCommandAllocator = _device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	_frames[1].GraphicsCommandAllocator = _device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	_frames[2].GraphicsCommandAllocator = _device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);

	_graphicsCommandBuffer = std::make_unique<Vcl::Graphics::Runtime::D3D12::CommandBuffer>(_device, _frames[0].GraphicsCommandAllocator.Get());

	return true;
}

void Application::resetCommandList()
{
	_graphicsCommandBuffer->reset(_frames[_swapChain->currentBufferIndex()].GraphicsCommandAllocator.Get());
}

LRESULT WINAPI Application::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	auto self = reinterpret_cast<Application*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
	if (self && self->msgHandler(hWnd, msg, wParam, lParam))
		return true;

	switch (msg)
	{
	case WM_CREATE:
	{
		auto* create_struct = reinterpret_cast<CREATESTRUCT*>(lParam);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create_struct->lpCreateParams));
		return 0;
	}
	case WM_SIZE:
		if (self && self->_device != nullptr && wParam != SIZE_MINIMIZED)
		{
			self->_swapChain->wait();
			self->invalidateDeviceObjects();
			self->_swapChain->resize(self->_device->defaultQueue(), (UINT)LOWORD(lParam), (UINT)HIWORD(lParam));
			self->createDeviceObjects();
		}
		return 0;
	case WM_SYSCOMMAND:
		if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
			return 0;
		break;
	case WM_DESTROY:
		::PostQuitMessage(0);
		return 0;
	}
	return ::DefWindowProc(hWnd, msg, wParam, lParam);
}
