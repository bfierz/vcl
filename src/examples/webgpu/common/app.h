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
#include <memory>

 // Windows Runtime Library
#define NOMINMAX
#include <tchar.h>
#include <windows.h>

// GLFW
#include <GLFW/glfw3.h>

// Dawn
#include <dawn_native/D3D12Backend.h>
#include <dawn_native/DawnNative.h>
#include <dawn/webgpu_cpp.h>

// VCL
#include <vcl/config/webgpu.h>

class Application
{
public:
	Application(LPCSTR title);
	virtual ~Application();

	GLFWwindow* windowHandle() const { return _windowHandle; }

	int run();

protected:
	bool initWebGpu(HWND hWnd);
	virtual void invalidateDeviceObjects();
	virtual void createDeviceObjects();
	virtual void updateFrame() {}
	virtual void renderFrame(WGPUTextureView back_buffer) {}

	//! Number of frames in the swap-queue
	static const int NumberOfFrames;

private:
	//! Resize the swapchain
	void resizeSwapChain(HWND hWnd, unsigned int width, unsigned int height);

	//! Handle to the GLFW window
	GLFWwindow* _windowHandle{ nullptr };

	//! Dawn WebGPU instance
	std::unique_ptr<dawn_native::Instance> _wgpuInstance;

	//! WebGPU surface
	WGPUSurface _wgpuSurface;
	
protected:
	//! WebGPU device
	WGPUDevice _wgpuDevice;

	//! SwapChain size
	std::pair<uint32_t, uint32_t> _swapChainSize;

private:
	//! WebGPU SwapChain
	WGPUSwapChain _swapChain;

	//! Dawn swap-chain
	DawnSwapChainImplementation _swapChainImpl = {};

	//! Swap-chain completion fence
	WGPUFence _swapChainFence;

	//! Frame presentation counter
	uint64_t _frameCounter = 0;
};
