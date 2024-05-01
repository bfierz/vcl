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

// VCL
#include <vcl/core/span.h>
#include <vcl/graphics/vulkan/descriptorset.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	class Context;

	class Semaphore
	{
	public:
		Semaphore();
		Semaphore(Context* context);
		Semaphore(Semaphore&& rhs);
		~Semaphore();

		Semaphore& operator=(Semaphore&& rhs) noexcept;

		//! Convert to Vulkan ID
		inline operator VkSemaphore() const
		{
			return _semaphore;
		}

		void wait();

	private:
		//! Owner
		Context* _context{ nullptr };

		//! Vulkan semaphore handle
		VkSemaphore _semaphore{ nullptr };
	};

	class Fence
	{
	public:
		Fence();
		Fence(Context* context, VkFenceCreateFlags flags);
		Fence(Fence&& rhs);
		~Fence();

		Fence& operator=(Fence&& rhs) noexcept;

		//! Convert to Vulkan ID
		inline operator VkFence() const
		{
			return _fence;
		}

		void reset();
		void wait();

	private:
		//! Owner
		Context* _context{ nullptr };

		//! Vulkan fence handle
		VkFence _fence{ nullptr };
	};

	class CommandBuffer
	{
	public:
		CommandBuffer() = default;
		CommandBuffer(VkDevice device, VkCommandPool pool);
		CommandBuffer(CommandBuffer&& rhs) noexcept;
		~CommandBuffer();

		CommandBuffer& operator=(CommandBuffer&& rhs) noexcept;

		//! Convert to Vulkan ID
		inline operator VkCommandBuffer() const
		{
			return _cmdBuffer;
		}

		void reset();

		//! \name General commands
		//! \{
		//! Begin recording commands
		void begin();

		//! End recording commands
		void end();

		//! Bind a pipeline
		void bind(VkPipeline pipeline);

		//! Bind a descriptor set
		void bind(VkDescriptorSet descriptors, VkPipelineLayout layout);

		//! Fill a buffer with a small memory pattern
		void fillBuffer(VkBuffer dst_buffer, size_t offset, size_t size, uint32_t data);

		//! Fill the push constant buffer
		template<typename T, typename... Args>
		void pushConstants(VkPipelineLayout layout, Flags<Graphics::Vulkan::ShaderStage> stages, Args&&... args)
		{
			T params = {std::forward<Args>(args)...};
			pushConstants(layout, {stages, 0, sizeof(T)}, &params);
		}

		//! Fill the push constant buffer
		void pushConstants(VkPipelineLayout layout, const PushConstantDescriptor& desc, void* data);

		//! Copy data between buffer objects
		void copy(VkBuffer src, VkBuffer dst, stdext::span<const VkBufferCopy> regions);
		//! \}


		//! \name Compute commands
		//! \{

		//! Dispatch compute task
		void dispatch(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1);
		//! \}

		//! \name Rendering commands
		//! \{

		//! Begin a render pass
		void beginRenderPass(const VkRenderPassBeginInfo* pass_info);

		//! End the current render pass
		void endRenderPass();

		//! Prepare a framebuffer for presentation
		void prepareForPresent(VkImage img);

		//! Transition an image from presentation to color attachment
		void returnFromPresent(VkImage img);

		//! Set scissor rect
		void setScissor(uint32_t first, stdext::span<VkRect2D> rects);

		//! Set viewport
		void setViewport(uint32_t first, stdext::span<VkViewport> viewports);

		//! \}

	private:
		VkDevice _device{ VK_NULL_HANDLE };
		VkCommandPool _pool{ VK_NULL_HANDLE };
		VkCommandBuffer _cmdBuffer{ VK_NULL_HANDLE };
	};

	class CommandQueue final
	{
	public:
		//! Constructor
		CommandQueue(Context* context, VkQueueFlags flags, unsigned int family_index);

		//! Destructor
		~CommandQueue();

		//! Convert to Vulkan ID
		inline operator VkQueue() const
		{
			return _queue;
		}

		unsigned int family() const
		{
			return _family_index;
		}

		void submit(
			const CommandBuffer& buffer,
			VkFence fence);

		void submit(
			const CommandBuffer& buffer,
			VkFence fence,
			VkPipelineStageFlags flags,
			VkSemaphore waiting,
			VkSemaphore signaling);
		
		void submit
		(
			stdext::span<VkCommandBuffer> buffers,
			VkFence fence,
			VkPipelineStageFlags flags = 0,
			stdext::span<VkSemaphore> waiting = {},
			stdext::span<VkSemaphore> signaling = {}
		);

		void waitIdle();

	private:
		//! Vulkan device queue
		VkQueue _queue;

		//! Queue family index
		unsigned int _family_index;
	};
}}}
