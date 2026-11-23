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

// C runtime
#include <cstring>

// C++ standard library
#include <iostream>
#include <string>
#include <exception>
#include <stdexcept>

#ifndef VCL_ARCH_WEBASM
// GLFW
#	if defined(_WIN32)
#		define GLFW_EXPOSE_NATIVE_WIN32
#	else
#		define GLFW_EXPOSE_NATIVE_X11
#	endif
#	include <GLFW/glfw3native.h>
#else
// Emscripten
#	include <emscripten.h>
#	include <emscripten/html5.h>
#	include <emscripten/html5_webgpu.h>
#endif

// VCL

const int Application::NumberOfFrames = 3;

static void printGlfwError(int error, const char* description)
{
	fprintf(stdout, "Glfw Error %d: %s\n", error, description);
}

static void printDeviceError(WGPUDevice const*, WGPUErrorType errorType, WGPUStringView message, void*, void*)
{
	const char* error_type = "";
	switch (errorType)
	{
	case WGPUErrorType_Validation:
		error_type = "Validation";
		break;
	case WGPUErrorType_OutOfMemory:
		error_type = "Out of memory";
		break;
	case WGPUErrorType_Internal:
		error_type = "Internal";
		break;
	case WGPUErrorType_Unknown:
		error_type = "Unknown";
		break;
	default:
		error_type = "Unexpected";
	}

	std::cout << error_type << " error: " << std::string(message.data, message.length == WGPU_STRLEN ? strlen(message.data) : message.length) << std::endl;
}

Application::Application(const char* title)
{
	glfwSetErrorCallback(printGlfwError);
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	_windowHandle = glfwCreateWindow(1280, 720, title, NULL, NULL);

	if (!initWebGpu(_windowHandle))
	{
		if (_windowHandle)
			glfwDestroyWindow(_windowHandle);
		glfwTerminate();
		throw std::runtime_error("WebGPU failed to initialize");
	}
}

Application::~Application()
{
	if (_swapChain)
		_swapChain->wait();
	_swapChain.reset();

	if (_wgpuSurface)
		wgpuSurfaceRelease(_wgpuSurface);
	if (_wgpuDevice)
		wgpuDeviceRelease(_wgpuDevice);
	if (_wgpuAdapter)
		wgpuAdapterRelease(_wgpuAdapter);
	if (_wgpuInstance)
		wgpuInstanceRelease(_wgpuInstance);

	if (_windowHandle)
		glfwDestroyWindow(_windowHandle);
	glfwTerminate();
}

void Application::step()
{
	glfwPollEvents();

	int width, height;
	glfwGetFramebufferSize(_windowHandle, &width, &height);

	// Nothing to render into while the window is minimized
	if (width == 0 || height == 0)
		return;

	if (width != _swapChainSize.first || height != _swapChainSize.second)
	{
		resizeSwapChain(_windowHandle, width, height);
	}

	// Allow to update the state of objects before waiting for the GPU
	updateFrame();

	auto back_buffer = _swapChain->currentBackBuffer();
	if (!back_buffer)
		return;

	renderFrame(back_buffer);
	_swapChain->present(wgpuDeviceGetQueue(_wgpuDevice), false);
}

void mainLoop(void* self)
{
	reinterpret_cast<Application*>(self)->step();
}

int Application::run()
{
	// Show the window
	if (_windowHandle)
		glfwShowWindow(_windowHandle);

#ifdef VCL_ARCH_WEBASM
	emscripten_set_main_loop_arg(mainLoop, this, 0, false);
#else
	while (!glfwWindowShouldClose(_windowHandle))
		step();
#endif

	return EXIT_SUCCESS;
}

void Application::invalidateDeviceObjects()
{
}

void Application::createDeviceObjects()
{
}

bool Application::initWebGpu(GLFWwindow* window)
{
#ifdef VCL_ARCH_WEBASM
	_wgpuDevice = emscripten_webgpu_get_device();

	WGPUEmscriptenSurfaceSourceCanvasHTMLSelector html_surface_desc = {};
	html_surface_desc.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
	html_surface_desc.selector = { "#canvas", WGPU_STRLEN };

	WGPUSurfaceDescriptor surface_desc = {};
	surface_desc.nextInChain = &html_surface_desc.chain;

	// Use 'null' instance
	_wgpuSurface = wgpuInstanceCreateSurface(nullptr, &surface_desc);
#else
	// Waiting on futures requires the timed wait feature
	const WGPUInstanceFeatureName features[] = { WGPUInstanceFeatureName_TimedWaitAny };
	WGPUInstanceDescriptor inst_desc = {};
	inst_desc.requiredFeatureCount = 1;
	inst_desc.requiredFeatures = features;
	_wgpuInstance = wgpuCreateInstance(&inst_desc);
	if (!_wgpuInstance)
		return false;

#	if defined(_WIN32)
	WGPUSurfaceSourceWindowsHWND native_surface_desc = {};
	native_surface_desc.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
	native_surface_desc.hinstance = GetModuleHandle(nullptr);
	native_surface_desc.hwnd = glfwGetWin32Window(window);
#	else
	WGPUSurfaceSourceXlibWindow native_surface_desc = {};
	native_surface_desc.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
	native_surface_desc.display = glfwGetX11Display();
	native_surface_desc.window = glfwGetX11Window(window);
#	endif

	// The surface has to exist before the adapter, in order to select an adapter that can present to it
	WGPUSurfaceDescriptor surface_desc = {};
	surface_desc.nextInChain = &native_surface_desc.chain;
	_wgpuSurface = wgpuInstanceCreateSurface(_wgpuInstance, &surface_desc);
	if (!_wgpuSurface)
		return false;

	WGPURequestAdapterOptions adapter_options = {};
	adapter_options.compatibleSurface = _wgpuSurface;

	WGPURequestAdapterCallbackInfo adapter_callback = {};
	adapter_callback.mode = WGPUCallbackMode_WaitAnyOnly;
	adapter_callback.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView, void* userdata1, void*) {
		if (status == WGPURequestAdapterStatus_Success)
			*static_cast<WGPUAdapter*>(userdata1) = adapter;
	};
	adapter_callback.userdata1 = &_wgpuAdapter;

	WGPUFutureWaitInfo adapter_wait = { wgpuInstanceRequestAdapter(_wgpuInstance, &adapter_options, adapter_callback), false };
	wgpuInstanceWaitAny(_wgpuInstance, 1, &adapter_wait, UINT64_MAX);
	if (!_wgpuAdapter)
		return false;

	WGPUDeviceDescriptor device_desc = {};
	device_desc.uncapturedErrorCallbackInfo.callback = printDeviceError;

	WGPURequestDeviceCallbackInfo device_callback = {};
	device_callback.mode = WGPUCallbackMode_WaitAnyOnly;
	device_callback.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView, void* userdata1, void*) {
		if (status == WGPURequestDeviceStatus_Success)
			*static_cast<WGPUDevice*>(userdata1) = device;
	};
	device_callback.userdata1 = &_wgpuDevice;

	WGPUFutureWaitInfo device_wait = { wgpuAdapterRequestDevice(_wgpuAdapter, &device_desc, device_callback), false };
	wgpuInstanceWaitAny(_wgpuInstance, 1, &device_wait, UINT64_MAX);
	if (!_wgpuDevice)
		return false;
#endif

	return true;
}

void Application::resizeSwapChain(GLFWwindow* window, unsigned int width, unsigned int height)
{
	if (_swapChain)
		_swapChain->wait();

	invalidateDeviceObjects();
	_swapChain.reset();

	_swapChainSize = { width, height };

	Vcl::Graphics::WebGPU::SwapChainDescription swap_chain_desc = {};
	swap_chain_desc.Surface = _wgpuSurface;
	swap_chain_desc.NumberOfImages = NumberOfFrames;
	swap_chain_desc.ColourFormat = Vcl::Graphics::SurfaceFormat::R8G8B8A8_UNORM;
	swap_chain_desc.Width = width;
	swap_chain_desc.Height = height;
	swap_chain_desc.PresentMode = Vcl::Graphics::WebGPU::PresentMode::Fifo;
	swap_chain_desc.VSync = true;
	_swapChain = std::make_unique<Vcl::Graphics::WebGPU::SwapChain>(_wgpuDevice, swap_chain_desc);
	_swapChainFormat = _swapChain->colourFormat();

	createDeviceObjects();
}
