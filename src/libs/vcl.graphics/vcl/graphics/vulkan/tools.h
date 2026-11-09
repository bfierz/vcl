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
#include <vcl/core/enummap.h>
#include <vcl/core/span.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace Vulkan {
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
	inline void setImageLayout(
		VkCommandBuffer cmdbuffer,
		VkImage image,
		VkImageAspectFlags aspectMask,
		VkImageLayout oldImageLayout,
		VkImageLayout newImageLayout,
		VkImageSubresourceRange subresourceRange,
		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
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
	inline void setImageLayout(
		VkCommandBuffer cmdbuffer,
		VkImage image,
		VkImageAspectFlags aspectMask,
		VkImageLayout oldImageLayout,
		VkImageLayout newImageLayout,
		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
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
		// clang-format off
		static constexpr EnumMapping<AttachmentLoadOperation, VkAttachmentLoadOp> table[] = {
			{ AttachmentLoadOperation::Load,     VK_ATTACHMENT_LOAD_OP_LOAD },
			{ AttachmentLoadOperation::Clear,    VK_ATTACHMENT_LOAD_OP_CLEAR },
			{ AttachmentLoadOperation::DontCare, VK_ATTACHMENT_LOAD_OP_DONT_CARE },
		};
		// clang-format on
		static constexpr EnumMap<AttachmentLoadOperation, VkAttachmentLoadOp, enumMapSize(table)> ops{ table };

		return ops[op];
	}

	enum class AttachmentStoreOperation
	{
		Store,
		DontCare
	};
	inline VkAttachmentStoreOp convert(AttachmentStoreOperation op)
	{
		// clang-format off
		static constexpr EnumMapping<AttachmentStoreOperation, VkAttachmentStoreOp> table[] = {
			{ AttachmentStoreOperation::Store,    VK_ATTACHMENT_STORE_OP_STORE },
			{ AttachmentStoreOperation::DontCare, VK_ATTACHMENT_STORE_OP_DONT_CARE },
		};
		// clang-format on
		static constexpr EnumMap<AttachmentStoreOperation, VkAttachmentStoreOp, enumMapSize(table)> ops{ table };

		return ops[op];
	}

	struct AttachmentDescription
	{
		VkFormat Format;
		int NrSamples;
		AttachmentLoadOperation LoadOp;
		AttachmentStoreOperation StoreOp;
	};

	RenderPass createBasicRenderPass(
		VkDevice device,
		stdext::span<const AttachmentDescription> attachments);

	inline bool isDepthFormat(VkFormat fmt)
	{
		switch (fmt)
		{
		case VK_FORMAT_D16_UNORM:
		case VK_FORMAT_X8_D24_UNORM_PACK32:
		case VK_FORMAT_D32_SFLOAT:
		case VK_FORMAT_D16_UNORM_S8_UINT:
		case VK_FORMAT_D24_UNORM_S8_UINT:
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
			return true;
		default:
			return false;
		}
	}

	inline bool isStencilFormat(VkFormat fmt)
	{
		switch (fmt)
		{
		case VK_FORMAT_S8_UINT:
		case VK_FORMAT_D16_UNORM_S8_UINT:
		case VK_FORMAT_D24_UNORM_S8_UINT:
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
			return true;
		default:
			return false;
		}
	}

	inline VkFormat convert(Vcl::Graphics::SurfaceFormat fmt)
	{
		// Formats without an entry map to VK_FORMAT_UNDEFINED
		// clang-format off
		static constexpr EnumMapping<SurfaceFormat, VkFormat> table[] = {
			{ SurfaceFormat::R32G32B32A32_FLOAT,   VK_FORMAT_R32G32B32A32_SFLOAT },
			{ SurfaceFormat::R32G32B32A32_UINT,    VK_FORMAT_R32G32B32A32_UINT },
			{ SurfaceFormat::R32G32B32A32_SINT,    VK_FORMAT_R32G32B32A32_SINT },
			{ SurfaceFormat::R16G16B16A16_FLOAT,   VK_FORMAT_R16G16B16A16_SFLOAT },
			{ SurfaceFormat::R16G16B16A16_UNORM,   VK_FORMAT_R16G16B16A16_UNORM },
			{ SurfaceFormat::R16G16B16A16_UINT,    VK_FORMAT_R16G16B16A16_UINT },
			{ SurfaceFormat::R16G16B16A16_SNORM,   VK_FORMAT_R16G16B16A16_SNORM },
			{ SurfaceFormat::R16G16B16A16_SINT,    VK_FORMAT_R16G16B16A16_SINT },
			{ SurfaceFormat::R32G32B32_FLOAT,      VK_FORMAT_R32G32B32_SFLOAT },
			{ SurfaceFormat::R32G32B32_UINT,       VK_FORMAT_R32G32B32_UINT },
			{ SurfaceFormat::R32G32B32_SINT,       VK_FORMAT_R32G32B32_SINT },
			{ SurfaceFormat::R32G32_FLOAT,         VK_FORMAT_R32G32_SFLOAT },
			{ SurfaceFormat::R32G32_UINT,          VK_FORMAT_R32G32_UINT },
			{ SurfaceFormat::R32G32_SINT,          VK_FORMAT_R32G32_SINT },
			{ SurfaceFormat::D32_FLOAT_S8X24_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT },
			{ SurfaceFormat::R10G10B10A2_UNORM,    VK_FORMAT_A2B10G10R10_UNORM_PACK32 },
			{ SurfaceFormat::R10G10B10A2_UINT,     VK_FORMAT_A2B10G10R10_UINT_PACK32 },
			{ SurfaceFormat::R11G11B10_FLOAT,      VK_FORMAT_B10G11R11_UFLOAT_PACK32 },
			{ SurfaceFormat::R8G8B8A8_UNORM,       VK_FORMAT_R8G8B8A8_UNORM },
			{ SurfaceFormat::R8G8B8A8_UNORM_SRGB,  VK_FORMAT_R8G8B8A8_SRGB },
			{ SurfaceFormat::R8G8B8A8_UINT,        VK_FORMAT_R8G8B8A8_UINT },
			{ SurfaceFormat::R8G8B8A8_SNORM,       VK_FORMAT_R8G8B8A8_SNORM },
			{ SurfaceFormat::R8G8B8A8_SINT,        VK_FORMAT_R8G8B8A8_SINT },
			{ SurfaceFormat::R16G16_FLOAT,         VK_FORMAT_R16G16_SFLOAT },
			{ SurfaceFormat::R16G16_UNORM,         VK_FORMAT_R16G16_UNORM },
			{ SurfaceFormat::R16G16_UINT,          VK_FORMAT_R16G16_UINT },
			{ SurfaceFormat::R16G16_SNORM,         VK_FORMAT_R16G16_SNORM },
			{ SurfaceFormat::R16G16_SINT,          VK_FORMAT_R16G16_SINT },
			{ SurfaceFormat::D32_FLOAT,            VK_FORMAT_D32_SFLOAT },
			{ SurfaceFormat::R32_FLOAT,            VK_FORMAT_R32_SFLOAT },
			{ SurfaceFormat::R32_UINT,             VK_FORMAT_R32_UINT },
			{ SurfaceFormat::R32_SINT,             VK_FORMAT_R32_SINT },
			{ SurfaceFormat::D24_UNORM_S8_UINT,    VK_FORMAT_D24_UNORM_S8_UINT },
			{ SurfaceFormat::R8G8_UNORM,           VK_FORMAT_R8G8_UNORM },
			{ SurfaceFormat::R8G8_UINT,            VK_FORMAT_R8G8_UINT },
			{ SurfaceFormat::R8G8_SNORM,           VK_FORMAT_R8G8_SNORM },
			{ SurfaceFormat::R8G8_SINT,            VK_FORMAT_R8G8_SINT },
			{ SurfaceFormat::R16_FLOAT,            VK_FORMAT_R16_SFLOAT },
			{ SurfaceFormat::D16_UNORM,            VK_FORMAT_D16_UNORM },
			{ SurfaceFormat::R16_UNORM,            VK_FORMAT_R16_UNORM },
			{ SurfaceFormat::R16_UINT,             VK_FORMAT_R16_UINT },
			{ SurfaceFormat::R16_SNORM,            VK_FORMAT_R16_SNORM },
			{ SurfaceFormat::R16_SINT,             VK_FORMAT_R16_SINT },
			{ SurfaceFormat::R8_UNORM,             VK_FORMAT_R8_UNORM },
			{ SurfaceFormat::R8_UINT,              VK_FORMAT_R8_UINT },
			{ SurfaceFormat::R8_SNORM,             VK_FORMAT_R8_SNORM },
			{ SurfaceFormat::R8_SINT,              VK_FORMAT_R8_SINT },
		};
		// clang-format on
		static constexpr EnumMap<SurfaceFormat, VkFormat, enumMapSize(table)> formats{ table };

		return formats[fmt];
	}

	inline VkPrimitiveTopology convert(Vcl::Graphics::Runtime::PrimitiveType type)
	{
		using Vcl::Graphics::Runtime::PrimitiveType;

		// clang-format off
		static constexpr EnumMapping<PrimitiveType, VkPrimitiveTopology> table[] = {
			{ PrimitiveType::Pointlist,        VK_PRIMITIVE_TOPOLOGY_POINT_LIST },
			{ PrimitiveType::Linelist,         VK_PRIMITIVE_TOPOLOGY_LINE_LIST },
			{ PrimitiveType::Linestrip,        VK_PRIMITIVE_TOPOLOGY_LINE_STRIP },
			{ PrimitiveType::Trianglelist,     VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST },
			{ PrimitiveType::Trianglestrip,    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP },
			{ PrimitiveType::LinelistAdj,      VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY },
			{ PrimitiveType::LinestripAdj,     VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY },
			{ PrimitiveType::TrianglelistAdj,  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY },
			{ PrimitiveType::TrianglestripAdj, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY },
			{ PrimitiveType::Patch,            VK_PRIMITIVE_TOPOLOGY_PATCH_LIST },
		};
		// clang-format on
		static constexpr EnumMap<PrimitiveType, VkPrimitiveTopology, enumMapSize(table)> topologies{ table };

		return topologies[type];
	}
}}}
