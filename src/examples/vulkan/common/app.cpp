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
#include "app.h"

// C++ standard library
#include <exception>
#include <stdexcept>

// VCL
#include <vcl/core/contract.h>

const int Application::NumberOfFrames = 3;

Application::Application(const char* title)
{
	_glfw = std::make_unique<GlfwInstance>();
	if (!_glfw->isVulkanSupported())
	{
		throw std::runtime_error("Vulkan is not supported");
	}

	// Create a window
	_windowHandle = glfwCreateWindow(1280, 720, title, nullptr, nullptr);
	if (!_windowHandle)
	{
		throw std::runtime_error("Glfw failed to create window");
	}
	glfwSetWindowUserPointer(_windowHandle, this);

	if (!initVulkan(*_glfw, _windowHandle))
	{
		glfwDestroyWindow(_windowHandle);
		throw std::runtime_error("Vulkan failed to initialize");
	}
}

Application::~Application()
{
	_frames[0].frameFence = {};
	_frames[1].frameFence = {};
	_frames[2].frameFence = {};
	_frames[0].renderComplete = {};
	_frames[1].renderComplete = {};
	_frames[2].renderComplete = {};
	_frames[0].presentComplete = {};
	_frames[1].presentComplete = {};
	_frames[2].presentComplete = {};
	_frames[0].CommandBuffer = {};
	_frames[1].CommandBuffer = {};
	_frames[2].CommandBuffer = {};
	_frames[0].CommandPool = {};
	_frames[1].CommandPool = {};
	_frames[2].CommandPool = {};
	_graphicsCommandQueue.reset();

	_surface.reset();
	_context.reset();
	_platform.reset();

	glfwDestroyWindow(_windowHandle);
}

int Application::run()
{
	using namespace Vcl::Graphics::Vulkan;

	createDeviceObjects();

	bool rebuild_swap_chain = false;
	uint32_t curr_buf = 0;
	while (!glfwWindowShouldClose(windowHandle()))
	{
		glfwPollEvents();
		if (rebuild_swap_chain)
		{
			rebuild_swap_chain = false;
		}

		updateFrame();

		auto* swapChain = _surface->swapChain();

		// Wait for the previous frame to finish
		_frames[curr_buf].frameFence.wait();
		_frames[curr_buf].frameFence.reset();

		// Render the scene
		VkSemaphore curr_present_complete = _frames[curr_buf].presentComplete;
		uint32_t next_image;
		VkResult res = swapChain->acquireNextImage(curr_present_complete, VK_NULL_HANDLE, &next_image);
		if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
		{
			rebuild_swap_chain = true;
			_frames[curr_buf].frameFence.reset();
			continue;
		}
		VclCheck(curr_buf == next_image, "Image count is in sync");

		// Prepare the frame data for a new frame
		_frames[curr_buf].CommandBuffer.reset();
		_frames[curr_buf].CommandPool.reset();

		_frames[curr_buf].CommandBuffer.begin();
		_frames[curr_buf].CommandBuffer.returnFromPresent(swapChain->image(curr_buf));

		renderFrame(curr_buf, _graphicsCommandQueue.get(), swapChain->view(curr_buf), _surface->backbuffer()->depthBufferView());

		// Finish default command buffer
		_frames[curr_buf].CommandBuffer.prepareForPresent(swapChain->image(curr_buf));
		_frames[curr_buf].CommandBuffer.end();

		// Submit to the graphics queue
		VkPipelineStageFlags pipelineStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		VkSemaphore s0 = _frames[curr_buf].presentComplete;
		VkSemaphore s1 = _frames[curr_buf].renderComplete;
		VkCommandBuffer b0 = _frames[curr_buf].CommandBuffer;
		_graphicsCommandQueue->submit({ &b0, 1 }, _frames[curr_buf].frameFence, pipelineStages, { &s0, 1 }, { &s1, 1 });

		// Present the current buffer to the swap chain
		// We pass the signal semaphore from the submit info
		// to ensure that the image is not rendered until
		// all commands have been submitted
		res = _surface->swapChain()->queuePresent(*_graphicsCommandQueue, curr_buf, _frames[curr_buf].renderComplete);
		if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
		{
			rebuild_swap_chain = true;
		}

		curr_buf = (curr_buf + 1) % NumberOfFrames;
	}
	for (int i = 0; i < NumberOfFrames; i++)
	{
		_frames[curr_buf].frameFence.wait();
		_frames[curr_buf].frameFence.reset();
		_frames[curr_buf].CommandBuffer.reset();
		_frames[curr_buf].CommandPool.reset();
		curr_buf = (curr_buf + 1) % NumberOfFrames;
	}
	invalidateDeviceObjects();

	return EXIT_SUCCESS;
}

void Application::invalidateDeviceObjects()
{
}
void Application::createDeviceObjects()
{
}

bool Application::initVulkan(const GlfwInstance& glfw, GLFWwindow* windowHandle)
{
	using namespace Vcl::Graphics::Vulkan;

	// Vulkan extension
	std::array<const char*, 1> context_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	// Initialize the Vulkan platform
	_platform = std::make_unique<Platform>(glfw.vulkanExtensions());
	auto& device = _platform->device(0);
	_context = device.createContext(stdext::make_span(context_extensions));

	// Create a WSI surface for the window
	VkSurfaceKHR surface_ctx;
	glfwCreateWindowSurface(*_platform, windowHandle, nullptr, &surface_ctx);

	// Allocate a render queue for the application
	_graphicsCommandQueue = std::make_unique<CommandQueue>(_context.get(), VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT, 0);

	Vcl::Graphics::Vulkan::BasicSurfaceDescription desc;
	desc.Surface = surface_ctx;
	desc.NumberOfImages = 3;
	desc.ColourFormat = VK_FORMAT_B8G8R8A8_UNORM;
	desc.DepthFormat = VK_FORMAT_D32_SFLOAT;
	_surface = createBasicSurface(*_platform, *_context, *_graphicsCommandQueue, desc);

	_frames[0].frameFence = Fence{ _context.get(), VK_FENCE_CREATE_SIGNALED_BIT };
	_frames[1].frameFence = Fence{ _context.get(), VK_FENCE_CREATE_SIGNALED_BIT };
	_frames[2].frameFence = Fence{ _context.get(), VK_FENCE_CREATE_SIGNALED_BIT };
	_frames[0].renderComplete = Semaphore{ _context.get() };
	_frames[1].renderComplete = Semaphore{ _context.get() };
	_frames[2].renderComplete = Semaphore{ _context.get() };
	_frames[0].presentComplete = Semaphore{ _context.get() };
	_frames[1].presentComplete = Semaphore{ _context.get() };
	_frames[2].presentComplete = Semaphore{ _context.get() };
	_frames[0].CommandPool = CommandPool(*_context, 0);
	_frames[1].CommandPool = CommandPool(*_context, 0);
	_frames[2].CommandPool = CommandPool(*_context, 0);
	_frames[0].CommandBuffer = CommandBuffer(*_context, _frames[0].CommandPool[CommandBufferType::Default]);
	_frames[1].CommandBuffer = CommandBuffer(*_context, _frames[1].CommandPool[CommandBufferType::Default]);
	_frames[2].CommandBuffer = CommandBuffer(*_context, _frames[2].CommandPool[CommandBufferType::Default]);

	return true;
}
