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

void Application::FrameContext::alloc(Vcl::Graphics::Vulkan::Context* context)
{
	using namespace Vcl::Graphics::Vulkan;

	FrameFence = Fence{ context, VK_FENCE_CREATE_SIGNALED_BIT };
	RenderComplete = Semaphore{ context };
	PresentComplete = Semaphore{ context };
	CmdPool = CommandPool(*context, 0);
	PrepareCmdBuffer = CommandBuffer(*context, CmdPool[CommandBufferType::Default]);
	PresentCmdBuffer = CommandBuffer(*context, CmdPool[CommandBufferType::Default]);
}

void Application::FrameContext::free()
{
	FrameFence = {};
	RenderComplete = {};
	PresentComplete = {};
	PrepareCmdBuffer = {};
	PresentCmdBuffer = {};
	CmdPool = {};
}

void Application::FrameContext::waitAndReset()
{
	FrameFence.wait();
	FrameFence.reset();
	PrepareCmdBuffer.reset();
	PresentCmdBuffer.reset();
	CmdPool.reset();
}

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

			for (auto& frame : _frames)
			{
				frame.waitAndReset();
			}
			invalidateDeviceObjects();
			createDeviceObjects();

			curr_buf = 0;
		}

		auto* swapChain = _surface->swapChain();

		// Render the scene
		auto& curr_frame = _frames[curr_buf];
		VkSemaphore curr_present_complete = curr_frame.PresentComplete;
		uint32_t next_image;
		VkResult res = swapChain->acquireNextImage(curr_present_complete, VK_NULL_HANDLE, &next_image);
		if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
		{
			rebuild_swap_chain = true;
			continue;
		}
		VclCheck(curr_buf == next_image, "Image count is in sync");

		updateFrame();

		// Wait for the previous frame to finish
		// and prepare the frame data for a new frame
		curr_frame.waitAndReset();

		// Bring the render-target into a valid state
		curr_frame.PrepareCmdBuffer.begin();
		curr_frame.PrepareCmdBuffer.returnFromPresent(swapChain->image(curr_buf));
		curr_frame.PrepareCmdBuffer.end();
		_graphicsCommandQueue->submit(curr_frame.PrepareCmdBuffer, VK_NULL_HANDLE);

		// Start the render buffer usable for the application
		curr_frame.PresentCmdBuffer.begin();
		renderFrame(curr_buf, _graphicsCommandQueue.get(), swapChain->view(curr_buf), _surface->backbuffer()->depthBufferView());

		// Finish default command buffer
		curr_frame.PresentCmdBuffer.prepareForPresent(swapChain->image(curr_buf));
		curr_frame.PresentCmdBuffer.end();

		// Submit to the graphics queue
		VkPipelineStageFlags pipelineStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		_graphicsCommandQueue->submit(curr_frame.PresentCmdBuffer, curr_frame.FrameFence, pipelineStages, curr_frame.PresentComplete, curr_frame.RenderComplete);

		// Present the current buffer to the swap chain
		// We pass the signal semaphore from the submit info
		// to ensure that the image is not rendered until
		// all commands have been submitted
		res = _surface->swapChain()->queuePresent(*_graphicsCommandQueue, curr_buf, curr_frame.RenderComplete);
		if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
		{
			rebuild_swap_chain = true;
		}

		curr_buf = (curr_buf + 1) % NumberOfFrames;
	}
	for (auto& frame : _frames)
	{
		frame.waitAndReset();
	}
	invalidateDeviceObjects();

	return EXIT_SUCCESS;
}

void Application::invalidateDeviceObjects()
{
	for (auto& frame : _frames)
	{
		frame.free();
	}

	_surface.reset();
}
void Application::createDeviceObjects()
{
	using namespace Vcl::Graphics::Vulkan;

	// Create a WSI surface for the window
	VkSurfaceKHR surface_ctx;
	glfwCreateWindowSurface(*_platform, windowHandle(), nullptr, &surface_ctx);

	Vcl::Graphics::Vulkan::BasicSurfaceDescription desc;
	desc.Surface = surface_ctx;
	desc.NumberOfImages = NumberOfFrames;
	desc.ColourFormat = VK_FORMAT_B8G8R8A8_UNORM;
	desc.DepthFormat = VK_FORMAT_D32_SFLOAT;
	_surface = createBasicSurface(*_platform, *_context, *_graphicsCommandQueue, desc);

	for (auto& frame : _frames)
	{
		frame.alloc(_context.get());
	}
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

	// Allocate a render queue for the application
	_graphicsCommandQueue = std::make_unique<CommandQueue>(_context.get(), VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT, 0);

	return true;
}
