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

// VCL configuration
#include <vcl/config/global.h>

// Google test
#include <gtest/gtest.h>

// Windows
#include <windows.h>

// Include the relevant parts from the library
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/d3d12/commandqueue.h>
#include <vcl/graphics/d3d12/swapchain.h>

extern std::unique_ptr<Vcl::Graphics::D3D12::Device> device;

class D3D12SwapChainTest : public testing::Test
{
public:
	void SetUp() override
	{
		WNDCLASS wc = { 0 };
		wc.lpfnWndProc = WndProc;
		wc.hInstance = GetModuleHandle(NULL);
		wc.hbrBackground = (HBRUSH)(COLOR_BACKGROUND);
		wc.lpszClassName = "D3D12WindowClass";
		wc.style = CS_OWNDC;
		RegisterClass(&wc);
		_window_handle = CreateWindowEx(0, wc.lpszClassName, "D3D12Window", 0, 0, 0, 0, 0, HWND_MESSAGE, 0, 0, 0);
	}

	void TearDown() override
	{
		CloseWindow(_window_handle);
	}

protected:
	static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
	{
		switch (message)
		{
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		return 0;
	}

	//! Native window handle of the test window
	HWND _window_handle;
};

TEST_F(D3D12SwapChainTest, CreateDestroy)
{
	using namespace Vcl::Graphics::D3D12;

	SwapChainDescription desc;
	desc.Surface = _window_handle;
	desc.NumberOfImages = 3;
	desc.ColourFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.Width = 512;
	desc.Height = 512;
	desc.PresentMode = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	desc.VSync = false;

	SwapChain swap_chain{ device.get(), device->defaultQueue(), desc };
}

TEST_F(D3D12SwapChainTest, CreateResizeDestroy)
{
	using namespace Vcl::Graphics::D3D12;

	SwapChainDescription desc;
	desc.Surface = _window_handle;
	desc.NumberOfImages = 3;
	desc.ColourFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.Width = 512;
	desc.Height = 512;
	desc.PresentMode = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	desc.VSync = false;

	SwapChain swap_chain{ device.get(), device->defaultQueue(), desc };
	swap_chain.resize(device->defaultQueue(), 1024, 1024);
}

TEST_F(D3D12SwapChainTest, PresentSync)
{
	using namespace Vcl::Graphics::D3D12;

	SwapChainDescription desc;
	desc.Surface = _window_handle;
	desc.NumberOfImages = 3;
	desc.ColourFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.Width = 512;
	desc.Height = 512;
	desc.PresentMode = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	desc.VSync = false;

	SwapChain swap_chain{ device.get(), device->defaultQueue(), desc };

	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);

	for (int i = 0; i < desc.NumberOfImages * 2; ++i)
	{
		swap_chain.waitForNextFrame();
		const auto rtv = swap_chain.prepareFrame(cmd_list.Get());
		float clearColor[] = { 1.0f, 0.0f, 1.0f, 1.0f };
		cmd_list->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
		swap_chain.present(device->defaultQueue(), cmd_list.Get(), true);

		cmd_allocator->Reset();
		cmd_list->Reset(cmd_allocator.Get(), nullptr);
	}
}
