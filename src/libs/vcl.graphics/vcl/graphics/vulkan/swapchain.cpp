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
#include <vcl/graphics/vulkan/swapchain.h>

// C++ standard library
#include <vector>

// FMT
#include <fmt/format.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/tools.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	Surface::Surface(VkInstance instance, VkPhysicalDevice device, unsigned int queue_family_index, VkSurfaceKHR surface)
	: _instance(instance)
	, _device(device)
	, _surface(surface)
	{
		VkResult res;

		VCL_VK_GET_INSTANCE_PROC(instance, vkGetPhysicalDeviceSurfaceSupportKHR);
		VCL_VK_GET_INSTANCE_PROC(instance, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
		VCL_VK_GET_INSTANCE_PROC(instance, vkGetPhysicalDeviceSurfaceFormatsKHR);
		VCL_VK_GET_INSTANCE_PROC(instance, vkGetPhysicalDeviceSurfacePresentModesKHR);

		// Check if surface supports output
		VkBool32 supported;
		res = vkGetPhysicalDeviceSurfaceSupportKHR(device, queue_family_index, surface, &supported);
		VclCheck(res == VK_SUCCESS, "Surface support is queried.");

		// Get list of supported surface formats
		uint32_t nr_formats;
		res = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &nr_formats, nullptr);

		std::vector<VkSurfaceFormatKHR> surface_formats(nr_formats);
		res = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &nr_formats, surface_formats.data());

		// If the only format is VK_FORMAT_UNDEFINED
		//if ((nr_formats == 1) && (surface_formats[0].format == VK_FORMAT_UNDEFINED))
		//{
		//	_colourFormat = VK_FORMAT_B8G8R8A8_UNORM;
		//	_colourSpace = surface_formats[0].colorSpace;
		//}
		//else
		//{
		//}

		// Query surface properties and formats
		VkSurfaceCapabilitiesKHR surface_caps;
		res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &surface_caps);
		VclCheck(res == VK_SUCCESS, "Surface capabilities are queried.");

		// Query available present modes
		uint32_t nr_present_modes;
		res = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &nr_present_modes, nullptr);
		VclCheck(res == VK_SUCCESS, "Number of present modes are queried.");
		
		std::vector<VkPresentModeKHR> present_modes(nr_present_modes);
		res = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &nr_present_modes, present_modes.data());
		VclCheck(res == VK_SUCCESS, "Present modes are queried.");
	}

	Surface::~Surface()
	{
		_backbuffer.reset();
		_swapChain.reset();

		if (_surface)
			vkDestroySurfaceKHR(_instance, _surface, nullptr);
	}

	void Surface::setSwapChain(std::unique_ptr<SwapChain> swap_chain)
	{
		_swapChain = std::move(swap_chain);
	}

	void Surface::setBackbuffer(std::unique_ptr<Backbuffer> buffer)
	{
		_backbuffer = std::move(buffer);
	}

	SwapChain::SwapChain(gsl::not_null<Context*> context, VkCommandBuffer cmd_buffer, const SwapChainDescription& desc)
	: _context(context)
	, _desc(desc)
	{
		VkResult res;

		VCL_VK_GET_DEVICE_PROC(*context, vkCreateSwapchainKHR);
		VCL_VK_GET_DEVICE_PROC(*context, vkDestroySwapchainKHR);
		VCL_VK_GET_DEVICE_PROC(*context, vkGetSwapchainImagesKHR);
		VCL_VK_GET_DEVICE_PROC(*context, vkAcquireNextImageKHR);
		VCL_VK_GET_DEVICE_PROC(*context, vkQueuePresentKHR);

		VkSwapchainCreateInfoKHR sc_create_info;
		sc_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		sc_create_info.pNext = nullptr;
		sc_create_info.surface = desc.Surface;
		sc_create_info.minImageCount = desc.NumberOfImages;
		sc_create_info.imageFormat = desc.ColourFormat;
		sc_create_info.imageColorSpace = desc.ColourSpace;
		sc_create_info.imageExtent = { desc.Width, desc.Height };
		sc_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		sc_create_info.preTransform = (VkSurfaceTransformFlagBitsKHR) desc.PreTransform;
		sc_create_info.imageArrayLayers = 1;
		sc_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		sc_create_info.queueFamilyIndexCount = 0;
		sc_create_info.pQueueFamilyIndices = nullptr;
		sc_create_info.presentMode = desc.PresentMode;
		sc_create_info.oldSwapchain = nullptr;//oldSwapchain;
		sc_create_info.clipped = true;
		sc_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		sc_create_info.flags = 0;

		res = vkCreateSwapchainKHR(*context, &sc_create_info, nullptr, &_swapchain);
		VclCheck(res == VK_SUCCESS, "Swap chain was created.");

		uint32_t nr_images = 0;
		res = vkGetSwapchainImagesKHR(*context, _swapchain, &nr_images, nullptr);
		VclCheck(res == VK_SUCCESS, "Number of images can be queried.");

		// Ask the implementation for the swap-chain images
		_images.resize(nr_images);
		res = vkGetSwapchainImagesKHR(*context, _swapchain, &nr_images, _images.data());
		VclCheck(res == VK_SUCCESS, "Images can be accessed.");

		// Create the image views for the swap-chain images
		_views.resize(nr_images);
		for (uint32_t i = 0; i < nr_images; i++)
		{
			VkImageViewCreateInfo colorAttachmentView = {};
			colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			colorAttachmentView.pNext = nullptr;
			colorAttachmentView.format = desc.ColourFormat;
			colorAttachmentView.components = {
				VK_COMPONENT_SWIZZLE_R,
				VK_COMPONENT_SWIZZLE_G,
				VK_COMPONENT_SWIZZLE_B,
				VK_COMPONENT_SWIZZLE_A
			};
			colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			colorAttachmentView.subresourceRange.baseMipLevel = 0;
			colorAttachmentView.subresourceRange.levelCount = 1;
			colorAttachmentView.subresourceRange.baseArrayLayer = 0;
			colorAttachmentView.subresourceRange.layerCount = 1;
			colorAttachmentView.viewType = VK_IMAGE_VIEW_TYPE_2D;
			colorAttachmentView.flags = 0;

			// Transform images from initial (undefined) to present layout
			setImageLayout
			(
				cmd_buffer,
				_images[i],
				VK_IMAGE_ASPECT_COLOR_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
			);

			colorAttachmentView.image = _images[i];

			res = vkCreateImageView(*_context, &colorAttachmentView, nullptr, &_views[i]);
			VclCheckEx(res == VK_SUCCESS, "Image view was created.", fmt::format("Image index: {}", i));
		}
	}

	SwapChain::~SwapChain()
	{
		for (uint32_t i = 0; i < _images.size(); i++)
			vkDestroyImageView(*_context, _views[i], nullptr);

		if (_swapchain)
			vkDestroySwapchainKHR(*_context, _swapchain, nullptr);
	}

	VkResult SwapChain::acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* currentBuffer)
	{
		return vkAcquireNextImageKHR(*_context, _swapchain, UINT64_MAX, presentCompleteSemaphore, (VkFence)nullptr, currentBuffer);
	}

	VkResult SwapChain::queuePresent(VkQueue queue, uint32_t currentBuffer)
	{
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.pNext = nullptr;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &_swapchain;
		presentInfo.pImageIndices = &currentBuffer;

		return vkQueuePresentKHR(queue, &presentInfo);
	}

	void SwapChain::queuePresent(VkQueue queue, uint32_t currentBuffer, VkSemaphore waitSemaphore)
	{
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.pNext = nullptr;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &_swapchain;
		presentInfo.pImageIndices = &currentBuffer;

		if (waitSemaphore != VK_NULL_HANDLE)
		{
			presentInfo.pWaitSemaphores = &waitSemaphore;
			presentInfo.waitSemaphoreCount = 1;
		}
		VkResult res = vkQueuePresentKHR(queue, &presentInfo);
		VclCheck(res == VK_SUCCESS, "Queue present was submitted successfully.");
	}

	Backbuffer::Backbuffer(SwapChain* swapchain, VkCommandBuffer cmd_buffer, uint32_t width, uint32_t height, VkFormat depth_format)
	: _swapchain(swapchain)
	{
		createDepthBuffer(cmd_buffer, width, height, depth_format);
		//createFramebuffers(pass, width, height);
	}

	Backbuffer::~Backbuffer()
	{
		vkDestroyImageView(*_swapchain->context(), _depthBufferView, nullptr);
		vkDestroyImage(*_swapchain->context(), _depthBufferImage, nullptr);
		vkFreeMemory(*_swapchain->context(), _depthBufferMemory, nullptr);
	}

	//void Backbuffer::createFramebuffers(VkRenderPass pass, uint32_t width, uint32_t height)
	//{
	//	VkImageView attachments[2];
	//
	//	// Depth/Stencil attachment is the same for all frame buffers
	//	attachments[1] = _depthBufferView;
	//
	//	VkFramebufferCreateInfo frameBufferCreateInfo = {};
	//	frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	//	frameBufferCreateInfo.pNext = nullptr;
	//	frameBufferCreateInfo.renderPass = pass;
	//	frameBufferCreateInfo.attachmentCount = 2;
	//	frameBufferCreateInfo.pAttachments = attachments;
	//	frameBufferCreateInfo.width = width;
	//	frameBufferCreateInfo.height = height;
	//	frameBufferCreateInfo.layers = 1;
	//
	//	// Create frame buffers for every swap chain image
	//	_framebuffers.resize(_swapchain->nrImages());
	//	for (uint32_t i = 0; i < _framebuffers.size(); i++)
	//	{
	//		attachments[0] = _swapchain->view(i);
	//		VkResult res = vkCreateFramebuffer(*_swapchain->context(), &frameBufferCreateInfo, nullptr, &_framebuffers[i]);
	//		VclCheckEx(res == VK_SUCCESS, "Framebuffer was created.", fmt::format("Framebuffer: {}", i));
	//	}
	//}

	void Backbuffer::createDepthBuffer(VkCommandBuffer cmd_buffer, uint32_t width, uint32_t height, VkFormat depth_format)
	{
		VkResult res;

		VkImageCreateInfo image = {};
		image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image.pNext = nullptr;
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = depth_format;
		image.extent = { width, height, 1 };
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		image.flags = 0;

		res = vkCreateImage(*_swapchain->context(), &image, nullptr, &_depthBufferImage);
		VclCheck(res == VK_SUCCESS, "Image was created.");

		VkMemoryAllocateInfo mem_alloc = {};
		mem_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		mem_alloc.pNext = nullptr;
		mem_alloc.memoryTypeIndex = 0;

		VkMemoryRequirements mem_reqs;
		vkGetImageMemoryRequirements(*_swapchain->context(), _depthBufferImage, &mem_reqs);
		mem_alloc.allocationSize = mem_reqs.size;

		Device* dev = _swapchain->context()->device();
		mem_alloc.memoryTypeIndex = dev->getMemoryTypeIndex(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		res = vkAllocateMemory(*_swapchain->context(), &mem_alloc, nullptr, &_depthBufferMemory);
		VclCheck(res == VK_SUCCESS, "Image memory allocated.");

		res = vkBindImageMemory(*_swapchain->context(), _depthBufferImage, _depthBufferMemory, 0);
		VclCheck(res == VK_SUCCESS, "Image was bound to memory.");

		setImageLayout
		(
			cmd_buffer,
			_depthBufferImage,
			VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		);

		VkImageViewCreateInfo view = {};
		view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view.pNext = nullptr;
		view.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view.format = depth_format;
		view.flags = 0;
		view.image = _depthBufferImage;
		view.subresourceRange = {};
		view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
		view.subresourceRange.baseMipLevel = 0;
		view.subresourceRange.levelCount = 1;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.layerCount = 1;
		
		res = vkCreateImageView(*_swapchain->context(), &view, nullptr, &_depthBufferView);
		VclCheck(res == VK_SUCCESS, "Image memory allocated.");
	}
	std::unique_ptr<Surface> createBasicSurface(Platform& platform, Context& context, CommandQueue& queue, const BasicSurfaceDescription& desc)
	{
		// Create the render surface
		auto surface = std::make_unique<Surface>(platform, *context.device(), 0, desc.Surface);

		// Allocate a command buffer to submit the image buffer creation
		CommandBuffer cmd_buffer{ context, context.commandPool(0, CommandBufferType::Default) };
		cmd_buffer.begin();

		// Create a swap-chain for the surface
		SwapChainDescription sc_desc;
		sc_desc.Surface = *surface;
		sc_desc.NumberOfImages = desc.NumberOfImages;
		sc_desc.ColourFormat = desc.ColourFormat;
		sc_desc.ColourSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
		sc_desc.Width = desc.Width;
		sc_desc.Height = desc.Height;
		sc_desc.PresentMode = VK_PRESENT_MODE_FIFO_KHR;
		sc_desc.PreTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;

		auto swapchain = std::make_unique<SwapChain>(&context, cmd_buffer, sc_desc);
		surface->setSwapChain(std::move(swapchain));

		// Create a back-buffer object for the surface
		auto backbuffer = std::make_unique<Backbuffer>(surface->swapChain(), cmd_buffer, desc.Width, desc.Height, desc.DepthFormat);
		surface->setBackbuffer(std::move(backbuffer));

		// Submit the image create to the driver
		cmd_buffer.end();
		queue.submit(cmd_buffer);
		queue.waitIdle();

		return surface;
	}
}}}
