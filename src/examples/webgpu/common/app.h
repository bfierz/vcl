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

#include <vcl/config/global.h>

// C++ Standard Library
#include <array>
#include <memory>

// GLFW
#include <GLFW/glfw3.h>

#ifndef VCL_ARCH_WEBASM
// Dawn
#	include <dawn/native/D3D12Backend.h>
#	include <dawn/native/DawnNative.h>
#endif

// WebGPU
#include <webgpu/webgpu_cpp.h>

// VCL
#include <vcl/config/webgpu.h>
#include <vcl/graphics/webgpu/swapchain.h>

class Application
{
public:
	Application(const char* title);
	virtual ~Application();

	GLFWwindow* windowHandle() const { return _windowHandle; }
	WGPUDevice device() const { return _wgpuDevice; }

	int run();
	void step();

protected:
	bool initWebGpu(GLFWwindow* window);
	virtual void invalidateDeviceObjects();
	virtual void createDeviceObjects();
	virtual void updateFrame() {}
	virtual void renderFrame(WGPUTextureView back_buffer, WGPUTextureView depth_buffer) {}

	//! Number of frames in the swap-queue
	static const int NumberOfFrames;

private:
	//! Select a GPU adapter for the this application
	static void requestAdapterCallback(WGPURequestAdapterStatus status, WGPUAdapter adapter, char const* message, void* userdata);

	//! Resize the swapchain
	void resizeSwapChain(GLFWwindow* window, unsigned int width, unsigned int height);

	//! Handle to the GLFW window
	GLFWwindow* _windowHandle{ nullptr };

	//! WebGPU instance
	WGPUInstance _wgpuInstance{ nullptr };

	//! WebGPU adapter
	WGPUAdapter _wgpuAdapter{ nullptr };

	//! WebGPU device
	WGPUDevice _wgpuDevice{ nullptr };

	//! WebGPU surface
	WGPUSurface _wgpuSurface{ nullptr };

	//! WebGPU SwapChain
	std::unique_ptr<Vcl::Graphics::WebGPU::SwapChain> _swapChain;

	//! Depth buffer
	WGPUTexture _depthBuffer{ nullptr };

	//! Render view on the depth buffer
	WGPUTextureView _depthBufferView{ nullptr };

protected:
	//! SwapChain size
	std::pair<uint32_t, uint32_t> _swapChainSize;
};
