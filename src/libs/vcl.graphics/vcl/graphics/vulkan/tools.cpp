/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <vcl/graphics/vulkan/tools.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	RenderPass createBasicRenderPass
	(
		VkDevice device,
		stdext::span<const AttachmentDescription> attachments
	)
	{
		std::vector<VkAttachmentDescription> vk_attachments;
		vk_attachments.reserve(attachments.size());

		std::vector<VkAttachmentReference> color_references;
		color_references.reserve(attachments.size());
		VkAttachmentReference depth_reference;

		for (const auto& attachment : attachments)
		{
			VclCheck(!isStencilFormat(convert(attachment.Format)), "Stencil formats not supported");

			const auto num_samples = [](int samples)
			{
				switch (samples)
				{
				case  1: return VK_SAMPLE_COUNT_1_BIT;
				case  2: return VK_SAMPLE_COUNT_2_BIT;
				case  4: return VK_SAMPLE_COUNT_4_BIT;
				case  8: return VK_SAMPLE_COUNT_8_BIT;
				case 16: return VK_SAMPLE_COUNT_16_BIT;
				case 32: return VK_SAMPLE_COUNT_32_BIT;
				case 64: return VK_SAMPLE_COUNT_64_BIT;
				}
				return VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM;
			};

			const bool is_depth = isDepthFormat(attachment.Format);
			VkAttachmentReference attachment_ref = {};
			attachment_ref.attachment = static_cast<uint32_t>(vk_attachments.size());
			if (is_depth)
			{
				attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
				depth_reference = attachment_ref;
			}
			else
			{
				attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
				color_references.emplace_back(attachment_ref);
			}

			VkAttachmentDescription vk_attachment;
			vk_attachment.flags = 0;
			vk_attachment.format = convert(attachment.Format);
			vk_attachment.samples = num_samples(attachment.NrSamples);
			vk_attachment.loadOp = convert(attachment.LoadOp);
			vk_attachment.storeOp = convert(attachment.StoreOp);
			vk_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			vk_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			vk_attachment.initialLayout = is_depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			vk_attachment.finalLayout = is_depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			vk_attachments.emplace_back(vk_attachment);
		}

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.flags = 0;
		subpass.inputAttachmentCount = 0;
		subpass.pInputAttachments = nullptr;
		subpass.colorAttachmentCount = static_cast<uint32_t>(color_references.size());
		subpass.pColorAttachments = color_references.data();
		subpass.pResolveAttachments = nullptr;
		subpass.pDepthStencilAttachment = &depth_reference;
		subpass.preserveAttachmentCount = 0;
		subpass.pPreserveAttachments = nullptr;

		VkRenderPassCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		info.pNext = nullptr;
		info.attachmentCount = static_cast<uint32_t>(vk_attachments.size());
		info.pAttachments = vk_attachments.data();
		info.subpassCount = 1;
		info.pSubpasses = &subpass;
		info.dependencyCount = 0;
		info.pDependencies = nullptr;

		VkRenderPass pass;
		VkResult res = vkCreateRenderPass(device, &info, nullptr, &pass);

		VclEnsure(res == VK_SUCCESS, "Render-pass was created.");
		return RenderPass{ device, pass };
	}
}}}
