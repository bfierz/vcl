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
#include <iostream>
#include <exception>
#include <stdexcept>

// Dawn
#include <dawn/dawn_proc.h>

// VCL

const int Application::NumberOfFrames = 3;

static void printDeviceError(WGPUErrorType errorType, const char* message, void*)
{
	const char* error_type = "";
	switch (errorType) {
	case WGPUErrorType_Validation:
		error_type = "Validation";
		break;
	case WGPUErrorType_OutOfMemory:
		error_type = "Out of memory";
		break;
	case WGPUErrorType_Unknown:
		error_type = "Unknown";
		break;
	case WGPUErrorType_DeviceLost:
		error_type = "Device lost";
		break;
	default:
		std::cout << "Unknown" << std::endl;
	}

	std::cout << error_type << " error: " << message << std::endl;
}


Application::Application(LPCSTR title)
{
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, _T("VCL D3D12 Example"), nullptr };
	::RegisterClassEx(&wc);
	_windowHandle = ::CreateWindow(wc.lpszClassName, title, WS_OVERLAPPEDWINDOW, 100, 100, 1280, 800, nullptr, nullptr, wc.hInstance, this);

	if (!initWebGpu(_windowHandle))
	{
		::DestroyWindow(_windowHandle);
		::UnregisterClass(wc.lpszClassName, wc.hInstance);
		throw std::runtime_error("D3D12 failed to initialize");
	}
}

Application::~Application()
{
	::DestroyWindow(_windowHandle);
	::UnregisterClass(_T("VCL WebGPU Example"), GetModuleHandle(nullptr));
}

int Application::run()
{
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

		auto back_buffer = wgpuSwapChainGetCurrentTextureView(_swapChain);
		renderFrame(back_buffer);
		wgpuSwapChainPresent(_swapChain);
		wgpuQueueSignal(wgpuDeviceGetDefaultQueue(_wgpuDevice), _swapChainFence, ++_frameCounter);
	}

	return EXIT_SUCCESS;
}

void Application::invalidateDeviceObjects()
{
}

void Application::createDeviceObjects()
{
}

bool Application::initWebGpu(HWND hWnd)
{
	_wgpuInstance = std::make_unique<dawn_native::Instance>();
#ifdef VCL_DEBUG
	_wgpuInstance->EnableBackendValidation(true);
#endif
	_wgpuInstance->DiscoverDefaultAdapters();
	dawn_native::Adapter adapter = _wgpuInstance->GetDefaultAdapter();
	_wgpuDevice = adapter.CreateDevice();

	DawnProcTable procs = dawn_native::GetProcs();
	dawnProcSetProcs(&procs);
	procs.deviceSetUncapturedErrorCallback(_wgpuDevice, printDeviceError, nullptr);

	WGPUSurfaceDescriptorFromWindowsHWND hwnd_surface_desc = {};
	hwnd_surface_desc.chain.sType = WGPUSType_SurfaceDescriptorFromWindowsHWND;
	hwnd_surface_desc.hinstance = nullptr;
	hwnd_surface_desc.hwnd = _windowHandle;

	WGPUSurfaceDescriptor surface_desc = {};
	surface_desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&hwnd_surface_desc);
	_wgpuSurface = wgpuInstanceCreateSurface(_wgpuInstance->Get(), &surface_desc);

	_swapChainImpl = dawn_native::d3d12::CreateNativeSwapChainImpl(_wgpuDevice, hWnd);

	WGPUSwapChainDescriptor desc;
	desc.implementation = reinterpret_cast<uint64_t>(&_swapChainImpl);
	_swapChain = wgpuDeviceCreateSwapChain(_wgpuDevice, nullptr, &desc);

	WGPUFenceDescriptor fence_desc = { nullptr, nullptr, 0 };
	_swapChainFence = wgpuQueueCreateFence(wgpuDeviceGetDefaultQueue(_wgpuDevice), &fence_desc);

	return true;
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
		if (self && self->_wgpuDevice != nullptr && wParam != SIZE_MINIMIZED)
		{
			// Wait for the last pending presentation
			while (wgpuFenceGetCompletedValue(self->_swapChainFence) < self->_frameCounter)
			{
				// Emulate a device tick...
				wgpuQueueSubmit(wgpuDeviceGetDefaultQueue(self->_wgpuDevice), 0, nullptr);
			}

			self->invalidateDeviceObjects();

			// Create new swap-chain for proper size
			wgpuSwapChainRelease(self->_swapChain);

			// Emulate a device tick...
			wgpuQueueSubmit(wgpuDeviceGetDefaultQueue(self->_wgpuDevice), 0, nullptr);

			self->_swapChainImpl = dawn_native::d3d12::CreateNativeSwapChainImpl(self->_wgpuDevice, hWnd);
			WGPUSwapChainDescriptor desc;
			desc.implementation = reinterpret_cast<uint64_t>(&self->_swapChainImpl);
			self->_swapChain = wgpuDeviceCreateSwapChain(self->_wgpuDevice, nullptr, &desc);

			RECT client_rect = {};
			GetClientRect(hWnd, &client_rect);
			self->_swapChainSize = { client_rect.right - client_rect.left, client_rect.bottom - client_rect.top };

			wgpuSwapChainConfigure
			(
				self->_swapChain,
				dawn_native::d3d12::GetNativeSwapChainPreferredFormat(&self->_swapChainImpl),
				WGPUTextureUsage_OutputAttachment,
				self->_swapChainSize.first, self->_swapChainSize.second
			);
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
