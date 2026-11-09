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
	PresentComplete = Semaphore{ context };
	CmdPool = CommandPool(*context, 0);
	PrepareCmdBuffer = CommandBuffer(*context, CmdPool[CommandBufferType::Default]);
	PresentCmdBuffer = CommandBuffer(*context, CmdPool[CommandBufferType::Default]);
}

void Application::FrameContext::free()
{
	FrameFence = {};
	PresentComplete = {};
	PrepareCmdBuffer = {};
	PresentCmdBuffer = {};
	CmdPool = {};
}

void Application::FrameContext::reset()
{
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
	uint32_t curr_frame = 0;
	while (!glfwWindowShouldClose(windowHandle()))
	{
		glfwPollEvents();
		if (rebuild_swap_chain)
		{
			rebuild_swap_chain = false;

			// Also waits for the presents, which no fence covers
			_graphicsCommandQueue->waitIdle();
			invalidateDeviceObjects();
			createDeviceObjects();

			curr_frame = 0;
		}

		auto* swapChain = _surface->swapChain();

		// Wait for the frame to finish, so its acquire semaphore and
		// command buffers can be reused
		auto& frame = _frames[curr_frame];
		frame.FrameFence.wait();

		// The presentation engine may return the images in any order.
		// A suboptimal swap-chain still delivers an image and signals the semaphore.
		uint32_t image;
		VkResult res = swapChain->acquireNextImage(frame.PresentComplete, VK_NULL_HANDLE, &image);
		if (res == VK_ERROR_OUT_OF_DATE_KHR)
		{
			rebuild_swap_chain = true;
			continue;
		}
		if (res == VK_SUBOPTIMAL_KHR)
			rebuild_swap_chain = true;
		else
			VclCheck(res == VK_SUCCESS, "Swap-chain image is acquired.");

		// Only reset the fence once work is submitted that signals it again
		frame.reset();

		updateFrame();

		// Bring the render-target into a valid state
		frame.PrepareCmdBuffer.begin();
		frame.PrepareCmdBuffer.returnFromPresent(swapChain->image(image));
		frame.PrepareCmdBuffer.end();

		// Start the render buffer usable for the application
		frame.PresentCmdBuffer.begin();
		renderFrame(curr_frame, image, _graphicsCommandQueue.get(), swapChain->view(image), _surface->backbuffer()->depthBufferView());

		// Finish default command buffer
		frame.PresentCmdBuffer.prepareForPresent(swapChain->image(image));
		frame.PresentCmdBuffer.end();

		// Wait for the acquired image before writing to it
		const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		std::array<VkCommandBuffer, 2> buffers = { frame.PrepareCmdBuffer, frame.PresentCmdBuffer };
		VkSemaphore wait = frame.PresentComplete;
		VkSemaphore signal = _renderComplete[image];
		_graphicsCommandQueue->submit({ buffers.data(), buffers.size() }, frame.FrameFence, wait_stage, { &wait, 1 }, { &signal, 1 });

		// Present once rendering is done
		res = swapChain->queuePresent(*_graphicsCommandQueue, image, _renderComplete[image]);
		if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
		{
			rebuild_swap_chain = true;
		}

		curr_frame = (curr_frame + 1) % NumberOfFrames;
	}
	_graphicsCommandQueue->waitIdle();
	invalidateDeviceObjects();

	return EXIT_SUCCESS;
}

void Application::invalidateDeviceObjects()
{
	for (auto& frame : _frames)
	{
		frame.free();
	}
	_renderComplete.clear();

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

	const uint32_t nr_images = _surface->swapChain()->nrImages();
	_renderComplete.reserve(nr_images);
	for (uint32_t i = 0; i < nr_images; i++)
	{
		_renderComplete.emplace_back(_context.get());
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
