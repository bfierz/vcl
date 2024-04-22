/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2024 Basil Fierz
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
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
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
		Vcl::Graphics::Vulkan::Fence frameFence;
		Vcl::Graphics::Vulkan::Semaphore presentComplete;
		Vcl::Graphics::Vulkan::Semaphore renderComplete;
		Vcl::Graphics::Vulkan::CommandPool CommandPool;
		Vcl::Graphics::Vulkan::CommandBuffer CommandBuffer;
	};

	Application(const char* title);
	virtual ~Application();

	GLFWwindow* windowHandle() const { return _windowHandle; }

	int run();

protected:
	virtual void invalidateDeviceObjects();
	virtual void createDeviceObjects();
	virtual void updateFrame() {}
	virtual void renderFrame(uint32_t frame, Vcl::Graphics::Vulkan::CommandQueue* cmd_buffer, VkImageView renderbuffer, VkImageView depthbuffer) {}

	Vcl::Graphics::Vulkan::Platform* platform() const { return _platform.get(); }
	Vcl::Graphics::Vulkan::Context* context() const { return _context.get(); }
	Vcl::Graphics::Vulkan::Surface* surface() const { return _surface.get(); }
	Vcl::Graphics::Vulkan::CommandQueue* queue() const { return _graphicsCommandQueue.get(); }

	FrameContext* frame(int i) { return &_frames[i]; }

	//! Number of frames in the swap-queue
	static const int NumberOfFrames;

private:
	bool initVulkan(const GlfwInstance& glfw, GLFWwindow* windowHandle);

	//! GLFW instance
	std::unique_ptr<GlfwInstance> _glfw;

	//! Handle to the GLFW window
	GLFWwindow* _windowHandle{ nullptr };

	//! Vulkan platorm
	std::unique_ptr<Vcl::Graphics::Vulkan::Platform> _platform;

	//! Abstraction of the render device
	std::unique_ptr<Vcl::Graphics::Vulkan::Context> _context;

	//! Surface used to display rendered images
	std::unique_ptr<Vcl::Graphics::Vulkan::Surface> _surface;

	std::unique_ptr<Vcl::Graphics::Vulkan::CommandQueue> _graphicsCommandQueue;

	std::array<FrameContext, 3> _frames;
};
