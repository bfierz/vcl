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
#include <vcl/core/span.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	template<typename Func>
	Func getInstanceProc(VkInstance inst, const char* name)
	{
		return reinterpret_cast<Func>(vkGetInstanceProcAddr(inst, name));
	}

	template<typename Func>
	Func getDeviceProc(VkDevice dev, const char* name)
	{
		return reinterpret_cast<Func>(vkGetDeviceProcAddr(dev, name));
	}

#define VCL_VK_GET_INSTANCE_PROC(instance, name) name = getInstanceProc<PFN_##name>(instance, #name)
#define VCL_VK_GET_DEVICE_PROC(device, name) name = getDeviceProc<PFN_##name>(device, #name)

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
		VkImageSubresourceRange subresourceRange,
		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
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

		// Put barrier inside setup command buffer
		vkCmdPipelineBarrier(
			cmdbuffer,
			srcStageMask,
			dstStageMask,
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
		VkImageLayout newImageLayout,
		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
	)
	{
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = aspectMask;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = 1;
		subresourceRange.layerCount = 1;
		setImageLayout(cmdbuffer, image, aspectMask, oldImageLayout, newImageLayout, subresourceRange, srcStageMask, dstStageMask);
	}

	class RenderPass
	{
	public:
		explicit RenderPass(VkDevice dev, VkRenderPass rp)
			: _device(dev)
			, _renderPass(rp)
		{
		}
		~RenderPass()
		{
			vkDestroyRenderPass(_device, _renderPass, nullptr);
		}

		operator VkRenderPass() const { return _renderPass; }

	private:
		//! Vulkan device
		VkDevice _device;

		//! Vulkan resource
		VkRenderPass _renderPass;
	};

	enum class AttachmentLoadOperation
	{
		Load,
		Clear,
		DontCare
	};
	inline VkAttachmentLoadOp convert(AttachmentLoadOperation op)
	{
		switch (op)
		{
		case AttachmentLoadOperation::Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
		case AttachmentLoadOperation::Clear: return VK_ATTACHMENT_LOAD_OP_CLEAR;
		case AttachmentLoadOperation::DontCare: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		}
		return VK_ATTACHMENT_LOAD_OP_LOAD;
	}

	enum class AttachmentStoreOperation
	{
		Store,
		DontCare
	};
	inline VkAttachmentStoreOp convert(AttachmentStoreOperation op)
	{
		switch (op)
		{
		case AttachmentStoreOperation::Store:    return VK_ATTACHMENT_STORE_OP_STORE;
		case AttachmentStoreOperation::DontCare: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
		}
		return VK_ATTACHMENT_STORE_OP_STORE;
	}

	struct AttachmentDescription
	{
		Vcl::Graphics::SurfaceFormat Format;
		int NrSamples;
		AttachmentLoadOperation LoadOp;
		AttachmentStoreOperation StoreOp;
	};
	
	RenderPass createBasicRenderPass
	(
		VkDevice device,
		stdext::span<const AttachmentDescription> attachments
	);

	inline bool isStencilFormat(VkFormat fmt)
	{
		switch (fmt)
		{
		case VK_FORMAT_S8_UINT:
		case VK_FORMAT_D16_UNORM_S8_UINT:
		case VK_FORMAT_D24_UNORM_S8_UINT:
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
			return true;
		}

		return false;
	}
	
	inline VkFormat convert(Vcl::Graphics::SurfaceFormat fmt)
	{
		switch (fmt)
		{
		case SurfaceFormat::R32G32B32A32_FLOAT  : return VK_FORMAT_R32G32B32A32_SFLOAT;
		case SurfaceFormat::R32G32B32A32_UINT   : return VK_FORMAT_R32G32B32A32_UINT;
		case SurfaceFormat::R32G32B32A32_SINT   : return VK_FORMAT_R32G32B32A32_SINT;
		case SurfaceFormat::R16G16B16A16_FLOAT  : return VK_FORMAT_R16G16B16A16_SFLOAT;
		case SurfaceFormat::R16G16B16A16_UNORM  : return VK_FORMAT_R16G16B16A16_UNORM;
		case SurfaceFormat::R16G16B16A16_UINT   : return VK_FORMAT_R16G16B16A16_UINT;
		case SurfaceFormat::R16G16B16A16_SNORM  : return VK_FORMAT_R16G16B16A16_SNORM;
		case SurfaceFormat::R16G16B16A16_SINT   : return VK_FORMAT_R16G16B16A16_SINT;
		case SurfaceFormat::R32G32B32_FLOAT     : return VK_FORMAT_R32G32B32_SFLOAT;
		case SurfaceFormat::R32G32B32_UINT      : return VK_FORMAT_R32G32B32_UINT;
		case SurfaceFormat::R32G32B32_SINT      : return VK_FORMAT_R32G32B32_SINT;
		case SurfaceFormat::R32G32_FLOAT        : return VK_FORMAT_R32G32_SFLOAT;
		case SurfaceFormat::R32G32_UINT         : return VK_FORMAT_R32G32_UINT;
		case SurfaceFormat::R32G32_SINT         : return VK_FORMAT_R32G32_SINT;
		case SurfaceFormat::D32_FLOAT_S8X24_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
		case SurfaceFormat::R10G10B10A2_UNORM   : return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
		case SurfaceFormat::R10G10B10A2_UINT    : return VK_FORMAT_A2B10G10R10_UINT_PACK32;
		case SurfaceFormat::R11G11B10_FLOAT     : return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
		case SurfaceFormat::R8G8B8A8_UNORM      : return VK_FORMAT_R8G8B8A8_UNORM;
		case SurfaceFormat::R8G8B8A8_UNORM_SRGB : return VK_FORMAT_R8G8B8A8_SRGB;
		case SurfaceFormat::R8G8B8A8_UINT       : return VK_FORMAT_R8G8B8A8_UINT;
		case SurfaceFormat::R8G8B8A8_SNORM      : return VK_FORMAT_R8G8B8A8_SNORM;
		case SurfaceFormat::R8G8B8A8_SINT       : return VK_FORMAT_R8G8B8A8_SINT;
		case SurfaceFormat::R16G16_FLOAT        : return VK_FORMAT_R16G16_SFLOAT;
		case SurfaceFormat::R16G16_UNORM        : return VK_FORMAT_R16G16_UNORM;
		case SurfaceFormat::R16G16_UINT         : return VK_FORMAT_R16G16_UINT;
		case SurfaceFormat::R16G16_SNORM        : return VK_FORMAT_R16G16_SNORM;
		case SurfaceFormat::R16G16_SINT         : return VK_FORMAT_R16G16_SINT;
		case SurfaceFormat::D32_FLOAT           : return VK_FORMAT_D32_SFLOAT;
		case SurfaceFormat::R32_FLOAT           : return VK_FORMAT_R32_SFLOAT;
		case SurfaceFormat::R32_UINT            : return VK_FORMAT_R32_UINT;
		case SurfaceFormat::R32_SINT            : return VK_FORMAT_R32_SINT;
		case SurfaceFormat::D24_UNORM_S8_UINT   : return VK_FORMAT_D24_UNORM_S8_UINT;
		case SurfaceFormat::R8G8_UNORM          : return VK_FORMAT_R8G8_UNORM;
		case SurfaceFormat::R8G8_UINT           : return VK_FORMAT_R8G8_UINT;
		case SurfaceFormat::R8G8_SNORM          : return VK_FORMAT_R8G8_SNORM;
		case SurfaceFormat::R8G8_SINT           : return VK_FORMAT_R8G8_SINT;
		case SurfaceFormat::R16_FLOAT           : return VK_FORMAT_R16_SFLOAT;
		case SurfaceFormat::D16_UNORM           : return VK_FORMAT_D16_UNORM;
		case SurfaceFormat::R16_UNORM           : return VK_FORMAT_R16_UNORM;
		case SurfaceFormat::R16_UINT            : return VK_FORMAT_R16_UINT;
		case SurfaceFormat::R16_SNORM           : return VK_FORMAT_R16_SNORM;
		case SurfaceFormat::R16_SINT            : return VK_FORMAT_R16_SINT;
		case SurfaceFormat::R8_UNORM            : return VK_FORMAT_R8_UNORM;
		case SurfaceFormat::R8_UINT             : return VK_FORMAT_R8_UINT;
		case SurfaceFormat::R8_SNORM            : return VK_FORMAT_R8_SNORM;
		case SurfaceFormat::R8_SINT             : return VK_FORMAT_R8_SINT;
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
