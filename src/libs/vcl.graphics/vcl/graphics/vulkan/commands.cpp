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
#include <vcl/graphics/vulkan/commands.h>

// C++ standard library
#include <iostream>
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/vulkan/context.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	Semaphore::Semaphore(Context* context)
	: _context(context)
	{
		VkSemaphoreCreateInfo semaphoreCreateInfo = {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		semaphoreCreateInfo.pNext = nullptr;

		VkResult res = vkCreateSemaphore(*_context, &semaphoreCreateInfo, nullptr, &_semaphore);
		VclCheck(res == VK_SUCCESS, "Semaphore was created.");
	}

	Semaphore::~Semaphore()
	{
		vkDestroySemaphore(*_context, _semaphore, nullptr);
	}

	CommandBuffer::CommandBuffer(VkDevice device, VkCommandPool pool)
	: _device(device)
	, _pool(pool)
	{
		VkCommandBufferAllocateInfo info;
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool = pool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 1;
		info.pNext = nullptr;

		VkResult res = vkAllocateCommandBuffers(device, &info, &_cmdBuffer);
		VclCheck(res == VK_SUCCESS, "Command buffer was created.");
	}

	CommandBuffer::~CommandBuffer()
	{
		vkFreeCommandBuffers(_device, _pool, 1, &_cmdBuffer);
	}

	void CommandBuffer::begin()
	{
		VkCommandBufferBeginInfo info;
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		info.pNext = nullptr;
		info.flags = 0;
		info.pInheritanceInfo = nullptr;

		VkResult res = vkBeginCommandBuffer(_cmdBuffer, &info);
		VclEnsure(res == VK_SUCCESS, "Command buffer recording began.");
	}

	void CommandBuffer::end()
	{
		VkResult res = vkEndCommandBuffer(_cmdBuffer);
		VclEnsure(res == VK_SUCCESS, "Command buffer recording ended.");
	}

	void CommandBuffer::bind(VkPipeline pipeline)
	{
		vkCmdBindPipeline(_cmdBuffer, VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	}

	void CommandBuffer::bind(VkDescriptorSet descriptors, VkPipelineLayout layout)
	{
		vkCmdBindDescriptorSets(_cmdBuffer, VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptors, 0, nullptr);
	}

	void CommandBuffer::fillBuffer(VkBuffer dst_buffer, size_t offset, size_t size, uint32_t data)
	{
		vkCmdFillBuffer(_cmdBuffer, dst_buffer, offset, size, data);
	}

	void CommandBuffer::pushConstants
	(
		VkPipelineLayout layout, const PushConstantDescriptor& desc, void* data
	)
	{
		vkCmdPushConstants(_cmdBuffer, layout, convert(desc.StageFlags), desc.Offset, desc.Size, data);
	}

	void CommandBuffer::copy(VkBuffer src, VkBuffer dst, stdext::span<const VkBufferCopy> regions)
	{
		vkCmdCopyBuffer(_cmdBuffer, src, dst, regions.size(), regions.data());
	}

	void CommandBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z)
	{
		vkCmdDispatch(_cmdBuffer, x, y, z);
	}

	void CommandBuffer::beginRenderPass(const VkRenderPassBeginInfo* pass_info)
	{
		vkCmdBeginRenderPass(_cmdBuffer, pass_info, VK_SUBPASS_CONTENTS_INLINE);
	}

	void CommandBuffer::endRenderPass()
	{
		vkCmdEndRenderPass(_cmdBuffer);
	}

	void CommandBuffer::prepareForPresent(VkImage img)
	{
		// Add a present memory barrier to the end of the command buffer
		// This will transform the frame buffer color attachment to a
		// new layout for presenting it to the windowing system integration 
		VkImageMemoryBarrier prePresentBarrier;
		prePresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		prePresentBarrier.pNext = nullptr;
		prePresentBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		prePresentBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		prePresentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		prePresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		prePresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		prePresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		prePresentBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		prePresentBarrier.image = img;

		VkImageMemoryBarrier *pMemoryBarrier = &prePresentBarrier;
		vkCmdPipelineBarrier
		(
			_cmdBuffer,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &prePresentBarrier
		);
	}

	void CommandBuffer::returnFromPresent(VkImage img)
	{
		// Add a post present image memory barrier
		// This will transform the frame buffer color attachment back
		// to it's initial layout after it has been presented to the
		// windowing system
		VkImageMemoryBarrier postPresentBarrier;
		postPresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		postPresentBarrier.pNext = nullptr;
		postPresentBarrier.srcAccessMask = 0;
		postPresentBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		postPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		postPresentBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		postPresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		postPresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		postPresentBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		postPresentBarrier.image = img;

		// Put post present barrier into command buffer
		vkCmdPipelineBarrier
		(
			_cmdBuffer,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &postPresentBarrier
		);
	}

	void CommandBuffer::setScissor(uint32_t first, stdext::span<VkRect2D> rects)
	{
		vkCmdSetScissor(_cmdBuffer, first, rects.size(), rects.data());
	}

	void CommandBuffer::setViewport(uint32_t first, stdext::span<VkViewport> viewports)
	{
		vkCmdSetViewport(_cmdBuffer, first, viewports.size(), viewports.data());
	}

	CommandQueue::CommandQueue(Context* context, VkQueueFlags flags, unsigned int family_index)
	: _queue(context->queue(flags, family_index))
	, _family_index(family_index)
	{

	}

	CommandQueue::~CommandQueue()
	{

	}

	void CommandQueue::submit
	(
		stdext::span<VkCommandBuffer> buffers,
		VkPipelineStageFlags flags,
		stdext::span<VkSemaphore> waiting,
		stdext::span<VkSemaphore> signaling
	)
	{
		VkSubmitInfo info;
		info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		info.pNext = nullptr;
		info.commandBufferCount = uint32_t(buffers.size());
		info.pCommandBuffers = buffers.data();
		info.waitSemaphoreCount = uint32_t(waiting.size());
		info.pWaitSemaphores = waiting.data();
		info.pWaitDstStageMask = &flags;
		info.signalSemaphoreCount = uint32_t(signaling.size());
		info.pSignalSemaphores = signaling.data();

		VkResult res = vkQueueSubmit(_queue, 1, &info, VK_NULL_HANDLE);
		VclCheck(res == VK_SUCCESS, "Command buffer recording began.");
	}

	void CommandQueue::submit
	(
		const CommandBuffer& buffer
	)
	{
		VkCommandBuffer buf = buffer;
		submit({ &buf, 1 });
	}

	void CommandQueue::waitIdle()
	{
		VkResult res = vkQueueWaitIdle(_queue);
		VclCheck(res == VK_SUCCESS, "Command buffer recording began.");
	}
}}}
