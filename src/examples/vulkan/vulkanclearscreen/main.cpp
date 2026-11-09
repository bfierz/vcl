/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <array>
#include <iostream>
#include <vector>

// Vulkan API
#include <vulkan/vulkan.h>

// GLFW
#include <GLFW/glfw3.h>

// VCL
#include <vcl/graphics/runtime/vulkan/resource/buffer.h>
#include <vcl/graphics/runtime/vulkan/resource/shader.h>
#include <vcl/graphics/vulkan/commands.h>
#include <vcl/graphics/vulkan/platform.h>
#include <vcl/graphics/vulkan/swapchain.h>
#include <vcl/graphics/vulkan/tools.h>

// Force the use of the NVIDIA GPU in an Optimius system
#ifdef VCL_COMPILER_MSVC
extern "C"
{
	_declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
}
#endif

void buildCommandBuffers(Vcl::Graphics::Vulkan::Backbuffer* bb, VkCommandPool cmdPool, VkRenderPass renderPass, stdext::span<const VkFramebuffer> framebuffers, uint32_t width, uint32_t height, std::vector<Vcl::Graphics::Vulkan::CommandBuffer>& drawCmdBuffers)
{
	drawCmdBuffers.reserve(bb->swapChain()->nrImages());
	for (int i = 0; i < bb->swapChain()->nrImages(); i++)
	{
		drawCmdBuffers.emplace_back(*bb->context(), cmdPool);
	}

	VkClearValue clearValues[2];
	clearValues[0].color = { 1.0f, 0.0f, 1.0f, 1.0f };
	clearValues[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.pNext = NULL;
	renderPassBeginInfo.renderPass = renderPass;
	renderPassBeginInfo.renderArea.offset.x = width / 4;
	renderPassBeginInfo.renderArea.offset.y = height / 4;
	renderPassBeginInfo.renderArea.extent.width = width / 2;
	renderPassBeginInfo.renderArea.extent.height = height / 2;
	renderPassBeginInfo.clearValueCount = 2;
	renderPassBeginInfo.pClearValues = clearValues;

	VkResult err;

	for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
	{
		// Set target frame buffer
		renderPassBeginInfo.framebuffer = framebuffers[i];

		drawCmdBuffers[i].begin();
		drawCmdBuffers[i].beginRenderPass(&renderPassBeginInfo);

		// Update dynamic viewport state
		VkViewport viewport = {};
		viewport.x = (float)width / 4;
		viewport.y = (float)height / 4;
		viewport.height = (float)height / 2;
		viewport.width = (float)width / 2;
		viewport.minDepth = (float)0.0f;
		viewport.maxDepth = (float)1.0f;
		drawCmdBuffers[i].setViewport(0, { &viewport, 1 });

		// Update dynamic scissor state
		VkRect2D scissor = {};
		scissor.extent.width = width / 2;
		scissor.extent.height = height / 2;
		scissor.offset.x = width / 4;
		scissor.offset.y = height / 4;
		drawCmdBuffers[i].setScissor(0, { &scissor, 1 });

		drawCmdBuffers[i].endRenderPass();

		// Preparing the frame buffer for presentation
		drawCmdBuffers[i].prepareForPresent(bb->swapChain()->image(i));

		// Finalizing the command buffer
		drawCmdBuffers[i].end();
	}
}

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

int main(int argc, char* argv[])
{
	using namespace Vcl::Graphics::Vulkan;

	GlfwInstance glfw;
	if (!glfw.isVulkanSupported())
	{
		std::cerr << "Vulkan is not supported" << std::endl;
		return 1;
	}

	// Vulkan extension
	std::array<const char*, 1> context_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	// Initialize the Vulkan platform
	auto platform = std::make_unique<Platform>(glfw.vulkanExtensions());
	auto& device = platform->device(0);
	auto context = device.createContext(stdext::make_span(context_extensions));
	CommandQueue queue{ context.get(), VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT, 0 };

	// Create a window
	auto window = glfwCreateWindow(1280, 720, "Vulkan Demo", nullptr, nullptr);
	if (!window)
	{
		std::cerr << "Cannot create a window in which to draw!" << std::endl;
		return 1;
	}

	// Create a WSI surface for the window
	VkSurfaceKHR surface_ctx;
	glfwCreateWindowSurface(*platform, window, nullptr, &surface_ctx);

	Vcl::Graphics::Vulkan::BasicSurfaceDescription desc;
	desc.Surface = surface_ctx;
	desc.NumberOfImages = 4;
	desc.ColourFormat = VK_FORMAT_B8G8R8A8_UNORM;
	desc.DepthFormat = VK_FORMAT_D32_SFLOAT;
	auto surface = createBasicSurface(*platform, *context, queue, desc);

	const std::array<AttachmentDescription, 2> attachments = {
		AttachmentDescription{ desc.ColourFormat, 1, AttachmentLoadOperation::Clear, AttachmentStoreOperation::Store },
		AttachmentDescription{ desc.DepthFormat, 1, AttachmentLoadOperation::Clear, AttachmentStoreOperation::Store },
	};
	auto render_pass = createBasicRenderPass(*context, attachments);

	// Allocate a framebuffer for each swap-chain image
	const uint32_t nr_images = surface->swapChain()->nrImages();
	std::vector<VkFramebuffer> framebuffers(nr_images);
	for (uint32_t i = 0; i < nr_images; i++)
	{
		framebuffers[i] = surface->backbuffer()->createFramebuffer(i, render_pass);
	}

	// Build command buffers
	std::vector<Vcl::Graphics::Vulkan::CommandBuffer> cmds;
	buildCommandBuffers(surface->backbuffer(), context->commandPool(0, CommandBufferType::Default), render_pass, framebuffers, 1280, 720, cmds);

	// Enter the event-loop. A single frame is in flight: the fence guards the
	// acquire semaphore and the post-present command buffer for reuse.
	CommandBuffer post_present{ *context, context->commandPool(0, CommandBufferType::Default) };
	Fence frame_complete{ context.get(), VK_FENCE_CREATE_SIGNALED_BIT };
	Semaphore present_complete{ context.get() };

	// The fence doesn't tell when the presentation engine has consumed the semaphore
	// a present waits on, so use one per swap-chain image. An image is only acquired
	// again once its previous present has finished with the semaphore.
	std::vector<Semaphore> render_complete;
	render_complete.reserve(nr_images);
	for (uint32_t i = 0; i < nr_images; i++)
		render_complete.emplace_back(context.get());

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		// Wait until the previous frame no longer uses the acquire semaphore
		frame_complete.wait();

		// A suboptimal swap-chain still delivers an image and signals the semaphore
		uint32_t curr_buf;
		const VkResult err = surface->swapChain()->acquireNextImage(present_complete, VK_NULL_HANDLE, &curr_buf);
		if (err != VK_SUCCESS && err != VK_SUBOPTIMAL_KHR)
			continue;

		frame_complete.reset();

		// Bring the image back from presentation
		post_present.begin();
		post_present.returnFromPresent(surface->swapChain()->image(curr_buf));
		post_present.end();

		// Wait for the acquired image before writing to it, and signal the
		// semaphore of this image once rendering is done
		const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		std::array<VkCommandBuffer, 2> buffers = { post_present, cmds[curr_buf] };
		VkSemaphore wait = present_complete;
		VkSemaphore signal = render_complete[curr_buf];
		queue.submit({ buffers.data(), buffers.size() }, frame_complete, wait_stage, { &wait, 1 }, { &signal, 1 });

		// Present once rendering is done
		surface->swapChain()->queuePresent(queue, curr_buf, render_complete[curr_buf]);
	}
	queue.waitIdle();

	for (auto& fb : framebuffers)
	{
		vkDestroyFramebuffer(*context, fb, nullptr);
	}

	return 0;
}
