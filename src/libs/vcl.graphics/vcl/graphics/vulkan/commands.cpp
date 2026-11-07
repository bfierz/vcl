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
#include <algorithm>
#include <iostream>
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/vulkan/resource/texture.h>
#include <vcl/graphics/vulkan/context.h>

namespace Vcl { namespace Graphics { namespace Vulkan {
	Semaphore::Semaphore() = default;

	Semaphore::Semaphore(Context* context)
	: _context(context)
	{
		VkSemaphoreCreateInfo semaphoreCreateInfo = {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		semaphoreCreateInfo.pNext = nullptr;

		VkResult res = vkCreateSemaphore(*_context, &semaphoreCreateInfo, nullptr, &_semaphore);
		VclCheck(res == VK_SUCCESS, "Semaphore was created.");
	}

	Semaphore::Semaphore(Semaphore&& rhs)
	{
		*this = std::move(rhs);
	}

	Semaphore::~Semaphore()
	{
		if (_semaphore != nullptr)
		{
			vkDestroySemaphore(*_context, _semaphore, nullptr);
		}
	}

	Semaphore& Semaphore::operator=(Semaphore&& rhs) noexcept
	{
		std::swap(_context, rhs._context);
		std::swap(_semaphore, rhs._semaphore);
		return *this;
	}

	void Semaphore::wait()
	{
		VkSemaphoreWaitInfo wait_info = {};
		wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
		wait_info.pNext = nullptr;
		wait_info.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
		wait_info.semaphoreCount = 1;
		wait_info.pSemaphores = &_semaphore;
		wait_info.pValues = nullptr;
		vkWaitSemaphores(*_context, &wait_info, UINT64_MAX);
	}

	Fence::Fence() = default;

	Fence::Fence(Context* context, VkFenceCreateFlags flags)
	: _context(context)
	{
		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.pNext = nullptr;
		fenceCreateInfo.flags = flags;

		VkResult res = vkCreateFence(*_context, &fenceCreateInfo, nullptr, &_fence);
		VclCheck(res == VK_SUCCESS, "Fence was created.");
	}

	Fence::Fence(Fence&& rhs)
	{
		*this = std::move(rhs);
	}

	Fence::~Fence()
	{
		if (_fence != nullptr)
		{
			vkDestroyFence(*_context, _fence, nullptr);
		}
	}

	Fence& Fence::operator=(Fence&& rhs) noexcept
	{
		std::swap(_context, rhs._context);
		std::swap(_fence, rhs._fence);
		return *this;
	}

	void Fence::reset()
	{
		vkResetFences(*_context, 1, &_fence);
	}

	void Fence::wait()
	{
		VkResult res = vkWaitForFences(*_context, 1, &_fence, VK_TRUE, UINT64_MAX);
		VclEnsure(res == VK_SUCCESS, "Fence was waited for.");
	}

	CommandBuffer::CommandBuffer(VkDevice device, VkCommandPool pool)
	: _device(device)
	, _pool(pool)
	{
		VkCommandBufferAllocateInfo info;
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.pNext = nullptr;
		info.commandPool = pool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 1;

		VkResult res = vkAllocateCommandBuffers(device, &info, &_cmdBuffer);
		VclEnsure(res == VK_SUCCESS, "Command buffer was created.");
	}

	CommandBuffer::CommandBuffer(CommandBuffer&& rhs) noexcept
	{
		*this = std::move(rhs);
	}
	CommandBuffer& CommandBuffer::operator=(CommandBuffer&& rhs) noexcept
	{
		std::swap(_device, rhs._device);
		std::swap(_pool, rhs._pool);
		std::swap(_cmdBuffer, rhs._cmdBuffer);
		return *this;
	}

	CommandBuffer::~CommandBuffer()
	{
		if (_cmdBuffer != nullptr)
		{
			vkFreeCommandBuffers(_device, _pool, 1, &_cmdBuffer);
			_cmdBuffer = nullptr;
		}
	}

	void CommandBuffer::reset()
	{
		vkResetCommandBuffer(_cmdBuffer, 0);
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

	void CommandBuffer::bind(VkPipelineBindPoint bind_point, VkPipeline pipeline)
	{
		vkCmdBindPipeline(_cmdBuffer, bind_point, pipeline);
	}

	void CommandBuffer::bind(VkPipelineBindPoint bind_point, VkDescriptorSet descriptors, VkPipelineLayout layout)
	{
		vkCmdBindDescriptorSets(_cmdBuffer, bind_point, layout, 0, 1, &descriptors, 0, nullptr);
	}

	void CommandBuffer::fillBuffer(VkBuffer dst_buffer, size_t offset, size_t size, uint32_t data)
	{
		vkCmdFillBuffer(_cmdBuffer, dst_buffer, offset, size, data);
	}

	void CommandBuffer::pushConstants(
		VkPipelineLayout layout,
		const PushConstantDescriptor& desc,
		void* data)
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

	namespace {
		VkAttachmentLoadOp toVulkan(Runtime::AttachmentLoadOp op)
		{
			switch (op)
			{
			case Runtime::AttachmentLoadOp::Clear: return VK_ATTACHMENT_LOAD_OP_CLEAR;
			case Runtime::AttachmentLoadOp::Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
			default: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			}
		}

		VkAttachmentStoreOp toVulkan(Runtime::AttachmentStoreOp op)
		{
			// 'Clear' discards the content, as in the D3D12 backend
			return op == Runtime::AttachmentStoreOp::Store ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
		}

		VkImageView toImageView(void* attachment)
		{
			static_assert(sizeof(VkImageView) == sizeof(void*), "Image view handle is compatible");
			return reinterpret_cast<VkImageView>(attachment);
		}
	}

	void CommandBuffer::beginRendering(const Runtime::RenderPassDescription& desc, const VkRect2D* render_area)
	{
		using Runtime::Vulkan::Texture;

		VkRect2D area = render_area ? *render_area : VkRect2D{};
		const auto update_area = [&area, render_area](const Texture* tex) {
			if (!render_area && area.extent.width == 0)
				area.extent = { static_cast<uint32_t>(tex->width()), static_cast<uint32_t>(tex->height()) };
		};

		std::vector<VkRenderingAttachmentInfo> colour_attachments;
		colour_attachments.reserve(desc.RenderTargetAttachments.size());
		for (const auto& rt : desc.RenderTargetAttachments)
		{
			VkRenderingAttachmentInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			if (rt.View)
			{
				auto tex = dynamic_cast<Texture*>(rt.View.get());
				VclRequire(tex, "Render-target is a Vulkan texture");
				tex->transition(_cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
				info.imageView = tex->view();
				update_area(tex);
			} else
			{
				info.imageView = toImageView(rt.Attachment);
			}
			info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			info.loadOp = toVulkan(rt.LoadOp);
			info.storeOp = toVulkan(rt.StoreOp);
			std::copy(rt.ClearColor.begin(), rt.ClearColor.end(), info.clearValue.color.float32);
			colour_attachments.push_back(info);
		}

		const auto& ds = desc.DepthStencilTargetAttachment;
		VkRenderingAttachmentInfo depth_attachment = {};
		VkRenderingAttachmentInfo stencil_attachment = {};
		bool has_depth = false;
		bool has_stencil = false;
		if (ds.View || ds.Attachment)
		{
			VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
			depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			if (ds.View)
			{
				auto tex = dynamic_cast<Texture*>(ds.View.get());
				VclRequire(tex, "Depth-stencil target is a Vulkan texture");
				tex->transition(_cmdBuffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
				depth_attachment.imageView = tex->view();
				aspects = tex->aspects();
				update_area(tex);
			} else
			{
				depth_attachment.imageView = toImageView(ds.Attachment);
			}
			depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			depth_attachment.loadOp = toVulkan(ds.DepthLoadOp);
			depth_attachment.storeOp = toVulkan(ds.DepthStoreOp);
			depth_attachment.clearValue.depthStencil = { ds.ClearDepth, ds.ClearStencil };
			has_depth = true;

			if (aspects & VK_IMAGE_ASPECT_STENCIL_BIT)
			{
				stencil_attachment = depth_attachment;
				stencil_attachment.loadOp = toVulkan(ds.StencilLoadOp);
				stencil_attachment.storeOp = toVulkan(ds.StencilStoreOp);
				has_stencil = true;
			}
		}

		VclCheck(area.extent.width > 0 && area.extent.height > 0, "Render area is valid.");

		VkRenderingInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		info.renderArea = area;
		info.layerCount = 1;
		info.colorAttachmentCount = static_cast<uint32_t>(colour_attachments.size());
		info.pColorAttachments = colour_attachments.data();
		info.pDepthAttachment = has_depth ? &depth_attachment : nullptr;
		info.pStencilAttachment = has_stencil ? &stencil_attachment : nullptr;
		vkCmdBeginRendering(_cmdBuffer, &info);
	}

	void CommandBuffer::endRendering()
	{
		vkCmdEndRendering(_cmdBuffer);
	}

	void CommandBuffer::bindVertexBuffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset)
	{
		vkCmdBindVertexBuffers(_cmdBuffer, binding, 1, &buffer, &offset);
	}

	void CommandBuffer::bindIndexBuffer(VkBuffer buffer, VkIndexType type, VkDeviceSize offset)
	{
		vkCmdBindIndexBuffer(_cmdBuffer, buffer, offset, type);
	}

	void CommandBuffer::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
	{
		vkCmdDraw(_cmdBuffer, vertex_count, instance_count, first_vertex, first_instance);
	}

	void CommandBuffer::drawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
	{
		vkCmdDrawIndexed(_cmdBuffer, index_count, instance_count, first_index, vertex_offset, first_instance);
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

		VkImageMemoryBarrier* pMemoryBarrier = &prePresentBarrier;
		vkCmdPipelineBarrier(
			_cmdBuffer,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &prePresentBarrier);
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
		vkCmdPipelineBarrier(
			_cmdBuffer,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &postPresentBarrier);
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

	void CommandQueue::submit(
		stdext::span<VkCommandBuffer> buffers,
		VkFence fence,
		VkPipelineStageFlags flags,
		stdext::span<VkSemaphore> waiting,
		stdext::span<VkSemaphore> signaling)
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

		VkResult res = vkQueueSubmit(_queue, 1, &info, fence);
		VclCheck(res == VK_SUCCESS, "Command buffer recording began.");
	}

	void CommandQueue::submit(
		const CommandBuffer& buffer,
		VkFence fence)
	{
		VkCommandBuffer buf = buffer;
		submit({ &buf, 1 }, fence);
	}

	void CommandQueue::submit(
		const CommandBuffer& buffer,
		VkFence fence,
		VkPipelineStageFlags flags,
		VkSemaphore waiting,
		VkSemaphore signaling)
	{
		VkCommandBuffer buf = buffer;
		submit({ &buf, 1 }, fence, flags, { &waiting, 1 }, { &signaling, 1 });
	}

	void CommandQueue::waitIdle()
	{
		VkResult res = vkQueueWaitIdle(_queue);
		VclCheck(res == VK_SUCCESS, "Command buffer recording began.");
	}
}}}
