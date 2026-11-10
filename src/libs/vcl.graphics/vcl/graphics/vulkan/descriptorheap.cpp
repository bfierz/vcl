/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2026 Basil Fierz
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
#include <vcl/graphics/vulkan/descriptorheap.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/vulkan/state/sampler.h>
#include <vcl/graphics/vulkan/commands.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/tools.h>

namespace Vcl { namespace Graphics { namespace Vulkan {
	using Runtime::ResourceIndex;
	using Runtime::ResourceViewType;
	using Runtime::SamplerIndex;

	namespace {
		//! Descriptor types a slot of the resource heap can hold
		const VkDescriptorType ResourceHeapTypes[] = {
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
			VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		};
		constexpr uint32_t NrResourceHeapTypes = sizeof(ResourceHeapTypes) / sizeof(VkDescriptorType);

		VkDescriptorType toVulkan(ResourceViewType type)
		{
			switch (type)
			{
			case ResourceViewType::ConstantBuffer:
				return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			case ResourceViewType::Buffer:
			case ResourceViewType::RWBuffer:
				return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			case ResourceViewType::Texture:
				return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			case ResourceViewType::RWTexture:
				return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			}

			VclDebugError("Unknown resource view type.");
			return VK_DESCRIPTOR_TYPE_MAX_ENUM;
		}

		bool isBufferView(ResourceViewType type)
		{
			return type == ResourceViewType::ConstantBuffer || type == ResourceViewType::Buffer || type == ResourceViewType::RWBuffer;
		}

		//! Mutable descriptor type lists of the heap set, and of the pool sizes
		//! in the same order: The resource heap followed by the sampler heap
		VkMutableDescriptorTypeCreateInfoEXT makeMutableTypeInfo(std::array<VkMutableDescriptorTypeListEXT, 2>& lists)
		{
			lists[0] = { NrResourceHeapTypes, ResourceHeapTypes };
			lists[1] = { 0, nullptr };

			VkMutableDescriptorTypeCreateInfoEXT info = {};
			info.sType = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT;
			info.mutableDescriptorTypeListCount = static_cast<uint32_t>(lists.size());
			info.pMutableDescriptorTypeLists = lists.data();
			return info;
		}
	}

	DescriptorHeap::DescriptorHeap(Context* context, const Runtime::BindlessLayoutDescription& desc)
	: _context(context)
	, _desc(desc)
	, _resources(desc.ResourceCapacity)
	, _samplers(desc.SamplerCapacity)
	, _viewTypes(desc.ResourceCapacity, ResourceViewType::ConstantBuffer)
	, _imageViews(desc.ResourceCapacity, VK_NULL_HANDLE)
	, _samplerObjects(desc.SamplerCapacity, VK_NULL_HANDLE)
	{
		VclRequire(context, "Context is set.");
		VclRequire(Runtime::validate(desc, context->features()).empty(), "Layout is supported by the device.");

		const bool anisotropy = context->enabledFeatures().samplerAnisotropy == VK_TRUE;
		VkResult res = VK_SUCCESS;

		// Heap set: Resource heap and sampler heap
		{
			VkDescriptorSetLayoutBinding bindings[2] = {};
			bindings[0].binding = ResourceHeapBinding;
			bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_MUTABLE_EXT;
			bindings[0].descriptorCount = desc.ResourceCapacity;
			bindings[0].stageFlags = VK_SHADER_STAGE_ALL;
			bindings[1].binding = SamplerHeapBinding;
			bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
			bindings[1].descriptorCount = desc.SamplerCapacity;
			bindings[1].stageFlags = VK_SHADER_STAGE_ALL;

			// Slots are filled on demand, also while the set is bound
			const VkDescriptorBindingFlags heap_flags =
				VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
				VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
				VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
			const VkDescriptorBindingFlags binding_flags[2] = { heap_flags, heap_flags };

			std::array<VkMutableDescriptorTypeListEXT, 2> type_lists;
			const VkMutableDescriptorTypeCreateInfoEXT mutable_info = makeMutableTypeInfo(type_lists);

			VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info = {};
			flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
			flags_info.pNext = &mutable_info;
			flags_info.bindingCount = 2;
			flags_info.pBindingFlags = binding_flags;

			VkDescriptorSetLayoutCreateInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			info.pNext = &flags_info;
			info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
			info.bindingCount = 2;
			info.pBindings = bindings;

			res = vkCreateDescriptorSetLayout(*_context, &info, nullptr, &_setLayouts[HeapSet]);
			VclCheck(res == VK_SUCCESS, "Heap set layout was created.");
		}

		// Static sampler set
		if (!desc.StaticSamplers.empty())
		{
			_staticSamplers.resize(desc.StaticSamplers.size(), VK_NULL_HANDLE);

			std::vector<VkDescriptorSetLayoutBinding> bindings(desc.StaticSamplers.size());
			for (size_t i = 0; i < desc.StaticSamplers.size(); i++)
			{
				const VkSamplerCreateInfo sampler_info = Runtime::Vulkan::toVulkan(desc.StaticSamplers[i].Sampler, anisotropy);
				res = vkCreateSampler(*_context, &sampler_info, nullptr, &_staticSamplers[i]);
				VclCheck(res == VK_SUCCESS, "Static sampler was created.");

				bindings[i] = {};
				bindings[i].binding = desc.StaticSamplers[i].Register;
				bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
				bindings[i].descriptorCount = 1;
				bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
				bindings[i].pImmutableSamplers = &_staticSamplers[i];
			}

			VkDescriptorSetLayoutCreateInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			info.bindingCount = static_cast<uint32_t>(bindings.size());
			info.pBindings = bindings.data();

			res = vkCreateDescriptorSetLayout(*_context, &info, nullptr, &_setLayouts[StaticSamplerSet]);
			VclCheck(res == VK_SUCCESS, "Static sampler set layout was created.");

			_nrSets = 2;
		}

		// Pipeline layout shared by all pipelines
		{
			const VkPushConstantRange push_constants = { VK_SHADER_STAGE_ALL, 0, desc.PushConstantSize };

			VkPipelineLayoutCreateInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			info.setLayoutCount = _nrSets;
			info.pSetLayouts = _setLayouts.data();
			info.pushConstantRangeCount = desc.PushConstantSize > 0 ? 1 : 0;
			info.pPushConstantRanges = &push_constants;

			res = vkCreatePipelineLayout(*_context, &info, nullptr, &_pipelineLayout);
			VclCheck(res == VK_SUCCESS, "Pipeline layout was created.");
		}

		// Descriptor sets
		{
			const VkDescriptorPoolSize pool_sizes[2] = {
				{ VK_DESCRIPTOR_TYPE_MUTABLE_EXT, desc.ResourceCapacity },
				{ VK_DESCRIPTOR_TYPE_SAMPLER, desc.SamplerCapacity + static_cast<uint32_t>(desc.StaticSamplers.size()) },
			};

			std::array<VkMutableDescriptorTypeListEXT, 2> type_lists;
			const VkMutableDescriptorTypeCreateInfoEXT mutable_info = makeMutableTypeInfo(type_lists);

			VkDescriptorPoolCreateInfo pool_info = {};
			pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			pool_info.pNext = &mutable_info;
			pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
			pool_info.maxSets = _nrSets;
			pool_info.poolSizeCount = 2;
			pool_info.pPoolSizes = pool_sizes;

			res = vkCreateDescriptorPool(*_context, &pool_info, nullptr, &_pool);
			VclCheck(res == VK_SUCCESS, "Descriptor pool was created.");

			VkDescriptorSetAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			alloc_info.descriptorPool = _pool;
			alloc_info.descriptorSetCount = _nrSets;
			alloc_info.pSetLayouts = _setLayouts.data();

			res = vkAllocateDescriptorSets(*_context, &alloc_info, _sets.data());
			VclEnsure(res == VK_SUCCESS, "Descriptor sets were allocated.");
		}
	}

	DescriptorHeap::~DescriptorHeap()
	{
		VclRequire(_resources.nrAllocated() == 0, "All resource views are released.");
		VclRequire(_samplers.nrAllocated() == 0, "All samplers are released.");

		for (auto view : _imageViews)
		{
			if (view)
				vkDestroyImageView(*_context, view, nullptr);
		}
		for (auto sampler : _samplerObjects)
		{
			if (sampler)
				vkDestroySampler(*_context, sampler, nullptr);
		}

		// Destroying the pool frees the sets
		if (_pool)
			vkDestroyDescriptorPool(*_context, _pool, nullptr);
		if (_pipelineLayout)
			vkDestroyPipelineLayout(*_context, _pipelineLayout, nullptr);
		for (auto layout : _setLayouts)
		{
			if (layout)
				vkDestroyDescriptorSetLayout(*_context, layout, nullptr);
		}
		for (auto sampler : _staticSamplers)
		{
			if (sampler)
				vkDestroySampler(*_context, sampler, nullptr);
		}
	}

	ResourceIndex DescriptorHeap::allocateResource(ResourceViewType type)
	{
		const ResourceIndex index{ _resources.allocate() };
		VclCheck(index.isValid(), "Resource heap has a free slot.");
		if (index.isValid())
			_viewTypes[index.Value] = type;

		return index;
	}

	ResourceView DescriptorHeap::createView(const Runtime::Vulkan::Buffer& buffer, ResourceViewType type, uint64_t offset, uint64_t size, uint32_t stride)
	{
		using Runtime::BufferUsage;

		VclRequire(isBufferView(type), "View type is a buffer type.");
		VclRequire(size > 0 && offset + size <= buffer.sizeInBytes(), "Range is inside the buffer.");
		VclRequire(implies(type == ResourceViewType::ConstantBuffer, buffer.usage().isSet(BufferUsage::Uniform)), "Buffer is usable as constant buffer.");
		VclRequire(implies(type != ResourceViewType::ConstantBuffer, buffer.usage().isSet(BufferUsage::Storage)), "Buffer is usable as storage buffer.");
		VclRequire(implies(stride > 0, offset % stride == 0 && size % stride == 0), "Range covers entire elements.");
		VCL_UNREFERENCED_PARAMETER(stride);

		const ResourceIndex index = allocateResource(type);
		if (!index.isValid())
			return {};

		const VkDescriptorBufferInfo buffer_info = { buffer.id(), offset, size };

		VkWriteDescriptorSet write = {};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = _sets[HeapSet];
		write.dstBinding = ResourceHeapBinding;
		write.dstArrayElement = index.Value;
		write.descriptorCount = 1;
		write.descriptorType = toVulkan(type);
		write.pBufferInfo = &buffer_info;
		vkUpdateDescriptorSets(*_context, 1, &write, 0, nullptr);

		return { this, index };
	}

	ResourceView DescriptorHeap::createView(const Runtime::Vulkan::Texture& texture, ResourceViewType type, const Runtime::TextureViewDescription& view)
	{
		using Runtime::TextureType;
		using Runtime::TextureUsage;
		using Runtime::TextureViewDescription;

		const bool read_write = type == ResourceViewType::RWTexture;
		const auto view_type = view.Type != TextureType::Unknown ? view.Type : texture.type();
		const uint32_t nr_mips = view.MipLevels != TextureViewDescription::Remaining ? view.MipLevels : static_cast<uint32_t>(texture.mipMapLevels()) - view.FirstMipLevel;
		const uint32_t nr_layers = view.Layers != TextureViewDescription::Remaining ? view.Layers : static_cast<uint32_t>(texture.layers()) - view.FirstLayer;

		VclRequire(!isBufferView(type), "View type is a texture type.");
		VclRequire(implies(!read_write, texture.usage().isSet(TextureUsage::Sampled)), "Texture is usable as sampled texture.");
		VclRequire(implies(read_write, texture.usage().isSet(TextureUsage::Storage)), "Texture is usable as storage texture.");
		VclRequire(nr_mips > 0 && view.FirstMipLevel + nr_mips <= static_cast<uint32_t>(texture.mipMapLevels()), "Mip-map levels are inside the texture.");
		VclRequire(nr_layers > 0 && view.FirstLayer + nr_layers <= static_cast<uint32_t>(texture.layers()), "Layers are inside the texture.");
		VclRequire(implies(read_write, nr_mips == 1), "Read-write views cover one mip-map level.");
		VclRequire(texture.supportsViewFormat(view.Format), "View format is the format of the texture or one of its view formats.");

		const ResourceIndex index = allocateResource(type);
		if (!index.isValid())
			return {};

		// A view of a depth-stencil texture used in a shader has a single aspect
		const VkImageAspectFlags aspects = (texture.aspects() & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : texture.aspects();

		VkImageViewCreateInfo view_info = {};
		view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_info.image = texture.id();
		view_info.viewType = Runtime::Vulkan::toVulkanImageViewType(view_type);
		view_info.format = convert(view.Format != SurfaceFormat::Unknown ? view.Format : texture.format());
		view_info.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
		view_info.subresourceRange = { aspects, view.FirstMipLevel, nr_mips, view.FirstLayer, nr_layers };

		// The slot can't hold a view anymore: it is destroyed before the slot is reused
		VclCheck(_imageViews[index.Value] == VK_NULL_HANDLE, "Slot is empty.");
		const VkResult res = vkCreateImageView(*_context, &view_info, nullptr, &_imageViews[index.Value]);
		VclCheck(res == VK_SUCCESS, "Image view was created.");

		VkDescriptorImageInfo image_info = {};
		image_info.imageView = _imageViews[index.Value];
		image_info.imageLayout = read_write ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkWriteDescriptorSet write = {};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = _sets[HeapSet];
		write.dstBinding = ResourceHeapBinding;
		write.dstArrayElement = index.Value;
		write.descriptorCount = 1;
		write.descriptorType = toVulkan(type);
		write.pImageInfo = &image_info;
		vkUpdateDescriptorSets(*_context, 1, &write, 0, nullptr);

		return { this, index };
	}

	SamplerView DescriptorHeap::createSampler(const Runtime::SamplerDescription& desc)
	{
		const SamplerIndex index{ _samplers.allocate() };
		VclCheck(index.isValid(), "Sampler heap has a free slot.");
		if (!index.isValid())
			return {};

		const VkSamplerCreateInfo sampler_info = Runtime::Vulkan::toVulkan(desc, _context->enabledFeatures().samplerAnisotropy == VK_TRUE);
		VclCheck(_samplerObjects[index.Value] == VK_NULL_HANDLE, "Slot is empty.");
		const VkResult res = vkCreateSampler(*_context, &sampler_info, nullptr, &_samplerObjects[index.Value]);
		VclCheck(res == VK_SUCCESS, "Sampler was created.");

		VkDescriptorImageInfo image_info = {};
		image_info.sampler = _samplerObjects[index.Value];

		VkWriteDescriptorSet write = {};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = _sets[HeapSet];
		write.dstBinding = SamplerHeapBinding;
		write.dstArrayElement = index.Value;
		write.descriptorCount = 1;
		write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
		write.pImageInfo = &image_info;
		vkUpdateDescriptorSets(*_context, 1, &write, 0, nullptr);

		return { this, index };
	}

	Runtime::ResourceViewType DescriptorHeap::viewType(ResourceIndex index) const
	{
		VclRequire(index.Value < _viewTypes.size(), "Index is valid.");
		return _viewTypes[index.Value];
	}

	void DescriptorHeap::retire(ResourceIndex index)
	{
		_resources.retire(index.Value);
	}

	void DescriptorHeap::retire(SamplerIndex index)
	{
		_samplers.retire(index.Value);
	}

	void DescriptorHeap::advance(uint64_t next_fence_value, uint64_t completed_fence_value)
	{
		// The GPU doesn't use the retired slots anymore. Their descriptors stay
		// in the set, which is valid as long as no shader accesses them.
		_resources.advance(next_fence_value, completed_fence_value, [this](uint32_t index) {
			if (_imageViews[index])
			{
				vkDestroyImageView(*_context, _imageViews[index], nullptr);
				_imageViews[index] = VK_NULL_HANDLE;
			}
		});
		_samplers.advance(next_fence_value, completed_fence_value, [this](uint32_t index) {
			vkDestroySampler(*_context, _samplerObjects[index], nullptr);
			_samplerObjects[index] = VK_NULL_HANDLE;
		});
	}

	void DescriptorHeap::bind(CommandBuffer& cmd_buffer) const
	{
		for (const auto bind_point : { VK_PIPELINE_BIND_POINT_GRAPHICS, VK_PIPELINE_BIND_POINT_COMPUTE })
			vkCmdBindDescriptorSets(cmd_buffer, bind_point, _pipelineLayout, 0, _nrSets, _sets.data(), 0, nullptr);
	}
}}}
