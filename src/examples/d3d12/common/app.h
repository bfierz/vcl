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

// C++ Standard Library
#include <array>

 // Windows Runtime Library
#define NOMINMAX
#include <tchar.h>
#include <wrl.h>

// VCL
#include <vcl/graphics/d3d12/device.h>
#include <vcl/graphics/d3d12/swapchain.h>
#include <vcl/graphics/runtime/d3d12/graphicsengine.h>

class Application
{
public:
	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;

	struct FrameContext
	{
		ComPtr<ID3D12CommandAllocator> GraphicsCommandAllocator;
	};

	Application(LPCSTR title);
	virtual ~Application();

	HWND windowHandle() const { return _windowHandle; }
	Vcl::Graphics::D3D12::Device* device() const { return _device.get(); }
	Vcl::Graphics::D3D12::SwapChain* swapChain() const{ return _swapChain.get(); }

	int run();

protected:
	virtual LRESULT msgHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) { return 0; }
	virtual void invalidateDeviceObjects();
	virtual void createDeviceObjects();
	virtual void updateFrame() {}
	virtual void renderFrame(ID3D12GraphicsCommandList*, D3D12_CPU_DESCRIPTOR_HANDLE rtv, D3D12_CPU_DESCRIPTOR_HANDLE dsv) {}
	virtual void renderFrame(Vcl::Graphics::Runtime::D3D12::CommandBuffer* cmd_buffer, D3D12_CPU_DESCRIPTOR_HANDLE rtv, D3D12_CPU_DESCRIPTOR_HANDLE dsv) {}

	ID3D12GraphicsCommandList* cmdList() const { return _graphicsCommandBuffer->handle(); }
	void resetCommandList();

	//! Number of frames in the swap-queue
	static const int NumberOfFrames;

private:
	bool initD3d12(HWND hWnd);
	static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

	//! Handle to the Win32 window
	HWND _windowHandle{ nullptr };

	//! Abstraction of the render device
	std::unique_ptr<Vcl::Graphics::D3D12::Device> _device;

	//! Swap-chain used to display rendered images
	std::unique_ptr<Vcl::Graphics::D3D12::SwapChain> _swapChain;

	ComPtr<ID3D12Resource> _depthBuffer;
	ComPtr<ID3D12DescriptorHeap> _dsvHeap;

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::CommandBuffer> _graphicsCommandBuffer;

	std::array<FrameContext, 3> _frames;
};
