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

// Vulkan
#include <vulkan/vulkan.h>

// GSL
#include <gsl/gsl>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	class Context;

	class Semaphore
	{
	public:
		Semaphore(Context* context);
		~Semaphore();

		//! Convert to Vulkan ID
		inline operator VkSemaphore() const
		{
			return _semaphore;
		}

	private:
		//! Owner
		Context* _context;

		//! Vulkan semaphore handle
		VkSemaphore _semaphore{ nullptr };
	};

	class CommandBuffer
	{
	public:
		CommandBuffer(VkDevice device, VkCommandPool pool);
		~CommandBuffer();

		//! Convert to Vulkan ID
		inline operator VkCommandBuffer() const
		{
			return _cmdBuffer;
		}

	public:
		void begin();
		void end();

		//! Begin a render pass
		void beginRenderPass(const VkRenderPassBeginInfo* pass_info);

		//! End the current render pass
		void endRenderPass();

		//! Prepare a framebuffer for presentation
		void prepareForPresent(VkImage img);

		//! Transition an image from presentation to color attachment
		void returnFromPresent(VkImage img);

		//! Set scissor rect
		void setScissor(uint32_t first, gsl::span<VkRect2D> rects);

		//! Set viewport
		void setViewport(uint32_t first, gsl::span<VkViewport> viewports);

	private:
		VkDevice _device;
		VkCommandPool _pool;
		VkCommandBuffer _cmdBuffer{ nullptr };
	};

	class CommandQueue final
	{
	public:
		//! Constructor
		CommandQueue(VkQueue queue);

		//! Destructor
		~CommandQueue();

		//! Convert to Vulkan ID
		inline operator VkQueue() const
		{
			return _queue;
		}

	public:
		void submit
		(
			const CommandBuffer& buffer
		);
		
		void submit
		(
			gsl::span<VkCommandBuffer> buffers,
			VkPipelineStageFlags* flags = nullptr,
			gsl::span<VkSemaphore> waiting = nullptr,
			gsl::span<VkSemaphore> signaling = nullptr
		);

		void waitIdle();

	private:
		//! Vulkan device queue
		VkQueue _queue;
	};
}}}
