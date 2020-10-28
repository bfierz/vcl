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

// GLFW
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

// Dawn
#include <dawn/dawn_proc.h>

// VCL

const int Application::NumberOfFrames = 3;

static void printGlfwError(int error, const char* description)
{
	fprintf(stdout, "Glfw Error %d: %s\n", error, description);
}

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
	glfwSetErrorCallback(printGlfwError);
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	_windowHandle = glfwCreateWindow(1280, 720, title, NULL, NULL);

	if (!initWebGpu(glfwGetWin32Window(_windowHandle)))
	{
		glfwDestroyWindow(_windowHandle);
		glfwTerminate();
		throw std::runtime_error("D3D12 failed to initialize");
	}
}

Application::~Application()
{
	glfwDestroyWindow(_windowHandle);
	glfwTerminate();
}

int Application::run()
{
	// Show the window
	glfwShowWindow(_windowHandle);

	while (!glfwWindowShouldClose(_windowHandle))
	{
		glfwPollEvents();

		int width, height;
		glfwGetFramebufferSize(_windowHandle, &width, &height);
		if (width != _swapChainSize.first && height != _swapChainSize.second)
		{
			resizeSwapChain(glfwGetWin32Window(_windowHandle), width, height);
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
	hwnd_surface_desc.hwnd = hWnd;

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

void Application::resizeSwapChain(HWND hWnd, unsigned int width, unsigned int height)
{
	// Wait for the last pending presentation
	while (wgpuFenceGetCompletedValue(_swapChainFence) < _frameCounter)
	{
		// Emulate a device tick...
		wgpuQueueSubmit(wgpuDeviceGetDefaultQueue(_wgpuDevice), 0, nullptr);
	}

	invalidateDeviceObjects();

	// Create new swap-chain for proper size
	wgpuSwapChainRelease(_swapChain);

	// Emulate a device tick...
	wgpuQueueSubmit(wgpuDeviceGetDefaultQueue(_wgpuDevice), 0, nullptr);

	_swapChainImpl = dawn_native::d3d12::CreateNativeSwapChainImpl(_wgpuDevice, hWnd);
	WGPUSwapChainDescriptor desc;
	desc.implementation = reinterpret_cast<uint64_t>(&_swapChainImpl);
	_swapChain = wgpuDeviceCreateSwapChain(_wgpuDevice, nullptr, &desc);

	_swapChainSize = { width, height };

	wgpuSwapChainConfigure
	(
		_swapChain,
		dawn_native::d3d12::GetNativeSwapChainPreferredFormat(&_swapChainImpl),
		WGPUTextureUsage_OutputAttachment,
		_swapChainSize.first, _swapChainSize.second
	);
	createDeviceObjects();
}
