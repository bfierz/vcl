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

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	// Taken from https://github.com/SaschaWillems/Vulkan/blob/master/base/vulkantools.cpp
	// Create an image memory barrier for changing the layout of
	// an image and put it into an active command buffer
	// See chapter 11.4 "Image Layout" for details
	inline void setImageLayout
	(
		VkCommandBuffer cmdbuffer,
		VkImage image,
		VkImageAspectFlags aspectMask,
		VkImageLayout oldImageLayout,
		VkImageLayout newImageLayout,
		VkImageSubresourceRange subresourceRange
	)
	{
		// Create an image barrier object
		VkImageMemoryBarrier imageMemoryBarrier;
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.pNext = nullptr;
		imageMemoryBarrier.srcAccessMask = 0;
		imageMemoryBarrier.dstAccessMask = 0;
		imageMemoryBarrier.oldLayout = oldImageLayout;
		imageMemoryBarrier.newLayout = newImageLayout;
		imageMemoryBarrier.srcQueueFamilyIndex = 0;
		imageMemoryBarrier.dstQueueFamilyIndex = 0;
		imageMemoryBarrier.image = image;
		imageMemoryBarrier.subresourceRange = subresourceRange;

		// Source layouts (old)

		// Undefined layout
		// Only allowed as initial layout!
		// Make sure any writes to the image have been finished
		if (oldImageLayout == VK_IMAGE_LAYOUT_PREINITIALIZED)
		{
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		}

		// Old layout is color attachment
		// Make sure any writes to the color buffer have been finished
		if (oldImageLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
		{
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		}

		// Old layout is depth/stencil attachment
		// Make sure any writes to the depth/stencil buffer have been finished
		if (oldImageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		}

		// Old layout is transfer source
		// Make sure any reads from the image have been finished
		if (oldImageLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
		{
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		}

		// Old layout is shader read (sampler, input attachment)
		// Make sure any shader reads from the image have been finished
		if (oldImageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}

		// Target layouts (new)

		// New layout is transfer destination (copy, blit)
		// Make sure any copyies to the image have been finished
		if (newImageLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		}

		// New layout is transfer source (copy, blit)
		// Make sure any reads from and writes to the image have been finished
		if (newImageLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
		{
			imageMemoryBarrier.srcAccessMask = imageMemoryBarrier.srcAccessMask | VK_ACCESS_TRANSFER_READ_BIT;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		}

		// New layout is color attachment
		// Make sure any writes to the color buffer hav been finished
		if (newImageLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
		{
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		}

		// New layout is depth attachment
		// Make sure any writes to depth/stencil buffer have been finished
		if (newImageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		}

		// New layout is shader read (sampler, input attachment)
		// Make sure any writes to the image have been finished
		if (newImageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}

		// Put barrier on top
		VkPipelineStageFlags srcStageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkPipelineStageFlags destStageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

		// Put barrier inside setup command buffer
		vkCmdPipelineBarrier(
			cmdbuffer,
			srcStageFlags,
			destStageFlags,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
	}

	// Taken from https://github.com/SaschaWillems/Vulkan/blob/master/base/vulkantools.cpp
	// Fixed sub resource on first mip level and layer
	inline void setImageLayout
	(
		VkCommandBuffer cmdbuffer,
		VkImage image,
		VkImageAspectFlags aspectMask,
		VkImageLayout oldImageLayout,
		VkImageLayout newImageLayout
	)
	{
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = aspectMask;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = 1;
		subresourceRange.layerCount = 1;
		setImageLayout(cmdbuffer, image, aspectMask, oldImageLayout, newImageLayout, subresourceRange);
	}

	inline VkRenderPass createDefaultRenderPass(VkDevice device, VkFormat color_format, VkFormat depth_format)
	{
		VkAttachmentDescription attachments[2];
		attachments[0].format = color_format;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		attachments[1].format = depth_format;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorReference = {};
		colorReference.attachment = 0;
		colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 1;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.flags = 0;
		subpass.inputAttachmentCount = 0;
		subpass.pInputAttachments = nullptr;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorReference;
		subpass.pResolveAttachments = nullptr;
		subpass.pDepthStencilAttachment = &depthReference;
		subpass.preserveAttachmentCount = 0;
		subpass.pPreserveAttachments = nullptr;

		VkRenderPassCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		info.pNext = nullptr;
		info.attachmentCount = 2;
		info.pAttachments = attachments;
		info.subpassCount = 1;
		info.pSubpasses = &subpass;
		info.dependencyCount = 0;
		info.pDependencies = nullptr;

		VkRenderPass pass;
		VkResult res = vkCreateRenderPass(device, &info, nullptr, &pass);
		VclCheck(res == VK_SUCCESS, "Render-pass was created.");

		return pass;
	}

	inline VkFormat convert(Vcl::Graphics::SurfaceFormat fmt)
	{
		switch (fmt)
		{
		case SurfaceFormat::R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
		case SurfaceFormat::R32G32B32A32_UINT   : return VK_FORMAT_R32G32B32A32_UINT;
		case SurfaceFormat::R32G32B32A32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
			//case SurfaceFormat::R16G16B16A16_FLOAT  : return RenderType<Eigen::Vector4f>();
			//case SurfaceFormat::R16G16B16A16_UNORM  : gl_format = GL_RGBA16; break;
			//case SurfaceFormat::R16G16B16A16_UINT   : return RenderType<Eigen::Vector4f>();
			//case SurfaceFormat::R16G16B16A16_SNORM  : gl_format = GL_RGBA16_SNORM; break;
			//case SurfaceFormat::R16G16B16A16_SINT   : return RenderType<Eigen::Vector4f>();
		case SurfaceFormat::R32G32B32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
		case SurfaceFormat::R32G32B32_UINT: return VK_FORMAT_R32G32B32_UINT;
		case SurfaceFormat::R32G32B32_SINT: return VK_FORMAT_R32G32B32_SINT;
		case SurfaceFormat::R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
		case SurfaceFormat::R32G32_UINT: return VK_FORMAT_R32G32_UINT;
		case SurfaceFormat::R32G32_SINT: return VK_FORMAT_R32G32_SINT;
			//case SurfaceFormat::D32_FLOAT_S8X24_UINT: gl_format = GL_DEPTH32F_STENCIL8; break;
			//case SurfaceFormat::R10G10B10A2_UNORM   : gl_format = GL_RGB10_A2; break;
			//case SurfaceFormat::R10G10B10A2_UINT    : gl_format = GL_RGB10_A2UI; break;
			//case SurfaceFormat::R11G11B10_FLOAT     : gl_format = GL_R11F_G11F_B10F; break;
			//case SurfaceFormat::R8G8B8A8_UNORM      : gl_format = GL_RGBA8; break;
			//case SurfaceFormat::R8G8B8A8_UNORM_SRGB : gl_format = GL_SRGB8_ALPHA8; break;
			//case SurfaceFormat::R8G8B8A8_UINT       : gl_format = GL_RGBA8UI; break;
			//case SurfaceFormat::R8G8B8A8_SNORM      : gl_format = GL_RGBA8_SNORM; break;
			//case SurfaceFormat::R8G8B8A8_SINT       : gl_format = GL_RGBA8I; break;
			//case SurfaceFormat::R16G16_FLOAT        : gl_format = GL_RG16F; break;
			//case SurfaceFormat::R16G16_UNORM        : gl_format = GL_RG16; break;
			//case SurfaceFormat::R16G16_UINT         : gl_format = GL_RG16UI; break;
			//case SurfaceFormat::R16G16_SNORM        : gl_format = GL_RG16_SNORM; break;
			//case SurfaceFormat::R16G16_SINT         : gl_format = GL_RG16I; break;
			//case SurfaceFormat::D32_FLOAT           : gl_format = GL_DEPTH_COMPONENT32F; break;
			//case SurfaceFormat::R32_FLOAT           : gl_format = GL_R32F; break;
			//case SurfaceFormat::R32_UINT            : gl_format = GL_R32UI; break;
			//case SurfaceFormat::R32_SINT            : gl_format = GL_R32I; break;
			//case SurfaceFormat::D24_UNORM_S8_UINT   : gl_format = GL_DEPTH24_STENCIL8; break;
			//case SurfaceFormat::R8G8_UNORM          : gl_format = GL_RG8; break;
			//case SurfaceFormat::R8G8_UINT           : gl_format = GL_RG8UI; break;
			//case SurfaceFormat::R8G8_SNORM          : gl_format = GL_RG8_SNORM; break;
			//case SurfaceFormat::R8G8_SINT           : gl_format = GL_RG8I; break;
			//case SurfaceFormat::R16_FLOAT           : gl_format = GL_R16F; break;
			//case SurfaceFormat::D16_UNORM           : gl_format = GL_DEPTH_COMPONENT16; break;
			//case SurfaceFormat::R16_UNORM           : gl_format = GL_R16; break;
			//case SurfaceFormat::R16_UINT            : gl_format = GL_R16UI; break;
			//case SurfaceFormat::R16_SNORM           : gl_format = GL_R16_SNORM; break;
			//case SurfaceFormat::R16_SINT            : gl_format = GL_R16I; break;
			//case SurfaceFormat::R8_UNORM            : gl_format = GL_R8; break;
			//case SurfaceFormat::R8_UINT             : gl_format = GL_R8UI; break;
			//case SurfaceFormat::R8_SNORM            : gl_format = GL_R8_SNORM; break;
			//case SurfaceFormat::R8_SINT             : gl_format = GL_R8I; break;
			default: VclDebugError("Unsupported colour format.");
		};

		return VK_FORMAT_UNDEFINED;
	}

	inline VkPrimitiveTopology convert(Vcl::Graphics::Runtime::PrimitiveType type)
	{
		using Vcl::Graphics::Runtime::PrimitiveType;

		switch (type)
		{
		case PrimitiveType::Pointlist:
			return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		case PrimitiveType::Linelist:
			return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		case PrimitiveType::Linestrip:
			return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
		case PrimitiveType::Trianglelist:
			return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		case PrimitiveType::Trianglestrip:
			return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
		case PrimitiveType::LinelistAdj:
			return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
		case PrimitiveType::LinestripAdj:
			return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
		case PrimitiveType::TrianglelistAdj:
			return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
		case PrimitiveType::TrianglestripAdj:
			return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
		case PrimitiveType::Patch:
			return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
		default: VclDebugError("Unsupported primitive type.");
		}

		return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
	}
}}}
