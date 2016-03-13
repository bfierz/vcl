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
#include <string>
#include <vector>

// Vulkan
#include <vulkan/vulkan.h>

// GSL
#include <gsl/gsl>

// VCL
#include <vcl/graphics/vulkan/context.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	class Surface final
	{
	public:
		/*!
		 *	\brief Constructor
		 *
		 *	\param surface Vulkan surface for which this swap-chain should be used
		 */
		Surface(VkInstance instance, VkPhysicalDevice device, VkSurfaceKHR surface);

		//! Destructor
		~Surface();

		//! Convert to Vulkan ID
		inline operator VkSurfaceKHR() const
		{
			return _surface;
		}

	private:
		//! Owner instance
		VkInstance _instance{ nullptr };
		
		//! Owner physical device
		VkPhysicalDevice _device{ nullptr };
		
		//! Surface of this swap chain
		VkSurfaceKHR _surface{ nullptr };

	private:
		PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR{ nullptr };
		PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR{ nullptr };
		PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR{ nullptr };
		PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR{ nullptr };
	};

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
		Backbuffer(SwapChain* swapchain, VkRenderPass pass, VkCommandBuffer cmd_buffer, uint32_t width, uint32_t height, VkFormat depth_format);
		~Backbuffer();

	public:
		Context* context() const { return _swapchain->context(); }

		SwapChain* swapChain() const { return _swapchain; }

		VkFramebuffer framebuffer(uint32_t idx) { return _framebuffers[idx]; }

	private:
		void createFramebuffers(VkRenderPass pass, uint32_t width, uint32_t height);
		void createDepthBuffer(VkCommandBuffer cmd_buffer, uint32_t width, uint32_t height, VkFormat depth_format);

	private:
		//! Owner
		SwapChain* _swapchain{ nullptr };

	private:
		//! Framebuffers
		std::vector<VkFramebuffer> _framebuffers;
		
	private: // Depth buffer
		//! Depth buffer memory
		VkDeviceMemory _depthBufferMemory;

		//! Depth buffer image
		VkImage _depthBufferImage;

		//! Depth buffer image view
		VkImageView _depthBufferView;
	};

}}}
