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

// VCL
#include <vcl/graphics/vulkan/commands.h>
#include <vcl/graphics/vulkan/platform.h>
#include <vcl/graphics/vulkan/swapchain.h>

// Vulkan API
#include <vulkan/vulkan.h>

// GLFW
#include <GLFW/glfw3.h>


class GlfwInstance
{
public:
	GlfwInstance()
	{
		glfwSetErrorCallback(errorCallback);
		_isInitialized = glfwInit() != 0 ? true : false;
		if (!_isInitialized)
			throw std::runtime_error{ "Unexpected error when initializing GLFW" };

		// Since we are using vulkan we do not need any predefined client API
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		// Check Vulkan support
		_vulkanSupport = glfwVulkanSupported() != 0 ? true : false;

		unsigned int nr_exts = 0;
		auto exts = glfwGetRequiredInstanceExtensions(&nr_exts);
		_vulkanExtensions = { exts, nr_exts };
	}

	~GlfwInstance()
	{
		if (_isInitialized)
			glfwTerminate();
	}

	//! \returns true if vulkan is supported
	bool isVulkanSupported() const { return _vulkanSupport; }

	//! \returns the required vulkan extensions
	stdext::span<const char*> vulkanExtensions() const { return _vulkanExtensions; }

private:
	static void errorCallback(int error, const char* description)
	{
		fprintf(stderr, "Error: %s\n", description);
	}

private:
	bool _isInitialized{ false };

	//! Query if Vulkan is supported
	bool _vulkanSupport{ false };

	//! Required Vulkan extensions
	stdext::span<const char*> _vulkanExtensions;
};


class Application
{
public:
	struct FrameContext
	{
	};

	Application(const char* title);
	virtual ~Application();

	GLFWwindow windowHandle() const { return _windowHandle; }
	Vcl::Graphics::Vulkan::Device* device() const { return _device.get(); }
	Vcl::Graphics::Vulkan::SwapChain* swapChain() const { return _swapChain.get(); }

	int run();

protected:
	virtual LRESULT msgHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) { return 0; }
	virtual void invalidateDeviceObjects();
	virtual void createDeviceObjects();
	virtual void updateFrame() {}
	virtual void renderFrame(Vcl::Graphics::Vulkan::CommandBuffer* cmd_buffer, D3D12_CPU_DESCRIPTOR_HANDLE rtv, D3D12_CPU_DESCRIPTOR_HANDLE dsv) {}

	ID3D12GraphicsCommandList* cmdList() const { return _graphicsCommandBuffer->handle(); }
	void resetCommandList();

	//! Number of frames in the swap-queue
	static const int NumberOfFrames;

private:
	bool initVulkan(HWND hWnd);

	//! Handle to the GLFW window
	GLFWwindow* _windowHandle{ nullptr };

	//! Abstraction of the render device
	std::unique_ptr<Vcl::Graphics::Vulkan::Device> _device;

	//! Swap-chain used to display rendered images
	std::unique_ptr<Vcl::Graphics::Vulkan::SwapChain> _swapChain;

	ComPtr<ID3D12Resource> _depthBuffer;
	ComPtr<ID3D12DescriptorHeap> _dsvHeap;

	std::unique_ptr<Vcl::Graphics::Vulkan::CommandBuffer> _graphicsCommandBuffer;

	std::array<FrameContext, 3> _frames;
};
