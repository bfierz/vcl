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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <memory>
#include <string>
#include <vector>

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/core/span.h>
#include <vcl/graphics/vulkan/commands.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/platform.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	struct SwapChainDescription
	{
		//! Handle to the surface used
		VkSurfaceKHR Surface;

		//! Number of images
		uint32_t NumberOfImages;

		//! Select colour format
		VkFormat ColourFormat;

		//! Select colour space
		VkColorSpaceKHR ColourSpace;

		//! Requested width
		uint32_t Width;

		//! Requested height
		uint32_t Height;

		//! Mode to present image
		VkPresentModeKHR PresentMode;

		//!
		VkSurfaceTransformFlagBitsKHR PreTransform;
	};

	class SwapChain final
	{
	public:
		//! Constructor
		SwapChain(Context* context, VkCommandBuffer cmd_buffer, const SwapChainDescription& desc);

		//! Destructor
		~SwapChain();

		//! Convert to Vulkan ID
		inline operator VkSwapchainKHR() const
		{
			return _swapchain;
		}

		Context* context() const { return _context; }

	public:
		//! Number of images in the swap-chain
		uint32_t nrImages() const { return _desc.NumberOfImages; }

		VkImage image(uint32_t idx) const { return _images[idx]; }
		VkImageView view(uint32_t idx) const { return _views[idx]; }

	public:
		//! Aquire the next image of the swap-chain
		VkResult acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* currentBuffer);

		//! Present the buffer to the queue
		VkResult queuePresent(VkQueue queue, uint32_t currentBuffer);

		//! Present the buffer to the queue
		void queuePresent(VkQueue queue, uint32_t currentBuffer, VkSemaphore waitSemaphore);


	private:
		//! Owner device
		Context* _context{ nullptr };

		//! Description
		SwapChainDescription _desc;

		//! Allocated swap-chain
		VkSwapchainKHR _swapchain{ nullptr };
		
		//! Swap chain images
		std::vector<VkImage> _images;

		//! Swap chain image views
		std::vector<VkImageView> _views;

	private:
		PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR{ nullptr };
		PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR{ nullptr };
		PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR{ nullptr };
		PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR{ nullptr };
		PFN_vkQueuePresentKHR vkQueuePresentKHR{ nullptr };
	};

	class Backbuffer final
	{
	public:
		Backbuffer(SwapChain* swapchain, VkCommandBuffer cmd_buffer, uint32_t width, uint32_t height, VkFormat depth_format);
		~Backbuffer();

	public:
		Context* context() const { return _swapchain->context(); }

		SwapChain* swapChain() const { return _swapchain; }

		//! Create a new frame buffer object for a swap-chain image
		VkFramebuffer createFramebuffer(uint32_t idx, VkRenderPass pass);

	private:
		void createDepthBuffer(VkCommandBuffer cmd_buffer, uint32_t width, uint32_t height, VkFormat depth_format);

	private:
		//! Owner
		SwapChain* _swapchain{ nullptr };

		//! Framebuffer width
		uint32_t _width;

		//! Framebuffer height
		uint32_t _height;

		//!{
		//! \name Depth buffer

		//! Depth buffer memory
		VkDeviceMemory _depthBufferMemory;

		//! Depth buffer image
		VkImage _depthBufferImage;

		//! Depth buffer image view
		VkImageView _depthBufferView;
		//!}
	};

	class Surface final
	{
	public:
		/*!
		*	\brief Constructor
		*
		*	\param surface Vulkan surface for which this swap-chain should be used
		*/
		Surface(VkInstance instance, VkPhysicalDevice device, unsigned int queue_family_index, VkSurfaceKHR surface);

		//! Destructor
		~Surface();

		//! Convert to Vulkan ID
		inline operator VkSurfaceKHR() const
		{
			return _surface;
		}

		//! Access the swap chain
		const SwapChain* swapChain() const { return _swapChain.get(); }

		//! Access the swap chain
		SwapChain* swapChain() { return _swapChain.get(); }

		//! Set the swap-chain
		void setSwapChain(std::unique_ptr<SwapChain> swap_chain);

		//! Access the default back-buffer
		Backbuffer* backbuffer() { return _backbuffer.get(); }

		//! Access the default back-buffer
		const Backbuffer* backbuffer() const { return _backbuffer.get(); }

		//! Set the default back-buffer
		void setBackbuffer(std::unique_ptr<Backbuffer> buffer);

	private:
		//! Owner instance
		VkInstance _instance{ nullptr };

		//! Owner physical device
		VkPhysicalDevice _device{ nullptr };

		//! Surface of this swap chain
		VkSurfaceKHR _surface{ nullptr };

		//! Swap-chain used to render to the surface
		std::unique_ptr<SwapChain> _swapChain;

		//! Default back-buffer
		std::unique_ptr<Backbuffer> _backbuffer;

	private:
		PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR{ nullptr };
		PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR{ nullptr };
		PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR{ nullptr };
		PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR{ nullptr };
	};

	struct BasicSurfaceDescription
	{
		//! Handle to the surface used
		VkSurfaceKHR Surface;

		//! Number of images
		uint32_t NumberOfImages;

		//! Select colour format
		VkFormat ColourFormat;

		//! Depth-format
		VkFormat DepthFormat;

		//! Requested width
		uint32_t Width;

		//! Requested height
		uint32_t Height;
	};

	std::unique_ptr<Surface> createBasicSurface(Platform& platform, Context& context, CommandQueue& queue, const BasicSurfaceDescription& desc);
}}}
