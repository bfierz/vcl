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
#include <vcl/graphics/vulkan/descriptorset.h>

// C++ standard library
#include <unordered_map>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	VkDescriptorType convert(DescriptorType type)
	{
		switch (type)
		{
		case DescriptorType::Sampler:              return VK_DESCRIPTOR_TYPE_SAMPLER;
		case DescriptorType::CombinedImageSampler: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		case DescriptorType::SampledImage:         return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		case DescriptorType::StorageImage:         return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		case DescriptorType::UniformTexelBuffer:   return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
		case DescriptorType::StorageTexelBuffer:   return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
		case DescriptorType::UniformBuffer:        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		case DescriptorType::StorageBuffer:        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		case DescriptorType::UniformBufferDynamic: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		case DescriptorType::StorageBufferDynamic: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
		case DescriptorType::InputAttachment:      return VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		}
		return VK_DESCRIPTOR_TYPE_MAX_ENUM;
	}

	VkShaderStageFlags convert(Flags<ShaderStage> stages)
	{
		VkShaderStageFlags flags = 0;
		flags |= stages.isSet(ShaderStage::Vertex) ? VK_SHADER_STAGE_VERTEX_BIT : 0;
		flags |= stages.isSet(ShaderStage::TessellationControl) ? VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT : 0;
		flags |= stages.isSet(ShaderStage::TessellationEvaluation) ? VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT : 0;
		flags |= stages.isSet(ShaderStage::Geometry) ? VK_SHADER_STAGE_GEOMETRY_BIT : 0;
		flags |= stages.isSet(ShaderStage::Fragment) ? VK_SHADER_STAGE_FRAGMENT_BIT : 0;
		flags |= stages.isSet(ShaderStage::Compute) ? VK_SHADER_STAGE_COMPUTE_BIT : 0;

		return flags;
	}

	std::vector<DescriptorSetLayoutBinding> merge
	(
		stdext::span<const DescriptorSetLayoutBinding> bindings0,
		stdext::span<const DescriptorSetLayoutBinding> bindings1
	)
	{
		VclRequire(std::is_sorted(std::begin(bindings0), std::end(bindings0),
			[](const DescriptorSetLayoutBinding& a, const DescriptorSetLayoutBinding& b) { return a.Binding < b.Binding; }),
			"'bindings0' are sorted according to the binding number");
		VclRequire(std::is_sorted(std::begin(bindings1), std::end(bindings1),
			[](const DescriptorSetLayoutBinding& a, const DescriptorSetLayoutBinding& b) { return a.Binding < b.Binding; }),
			"'bindings1' are sorted according to the binding number");

		std::vector<DescriptorSetLayoutBinding> bindings;
		bindings.reserve(bindings0.size() + bindings1.size());

		auto b0 = std::begin(bindings0);
		auto b1 = std::begin(bindings1);
		while (b0 != std::end(bindings0) && b1 != std::end(bindings1))
		{
			if (b0->Binding == b1->Binding)
			{
				if (b0->Type == b1->Type)
				{
					DescriptorSetLayoutBinding desc;
					desc.Type = b0->Type;
					desc.ShaderStages = b0->ShaderStages | b1->ShaderStages;
					desc.Binding = b0->Binding;
					bindings.emplace_back(desc);
				}
				else
				{
					return {};
				}

				++b0;
				++b1;
			}
			else if (b0->Binding < b1->Binding)
			{
				bindings.emplace_back(*b0);
				++b0;
			}
			else // (b0->Binding > b1->Binding)
			{
				bindings.emplace_back(*b1);
				++b1;
			}
		}
		if (b0 != std::end(bindings0))
		{
			bindings.insert(std::end(bindings), b0, std::end(bindings0));
		}
		if (b1 != std::end(bindings1))
		{
			bindings.insert(std::end(bindings), b1, std::end(bindings1));
		}

		return bindings;
	}

	DescriptorSetLayout::DescriptorSetLayout(Context* context, stdext::span<const DescriptorSetLayoutBinding> bindings)
		: _context(context)
		, _descriptor_summary(11, 0)
	{
		std::vector<VkDescriptorSetLayoutBinding> vk_bindings;
		vk_bindings.reserve(bindings.size());

		for (const auto& binding : bindings)
		{
			VkDescriptorSetLayoutBinding layout = {};
			layout.binding = binding.Binding;
			layout.descriptorType = convert(binding.Type);
			layout.descriptorCount = binding.Entries;
			layout.stageFlags = convert(binding.ShaderStages);
			layout.pImmutableSamplers = nullptr;
			vk_bindings.emplace_back(layout);

			_descriptor_summary[static_cast<size_t>(binding.Type)] += binding.Entries;
		}

		VkDescriptorSetLayoutCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		info.pNext = nullptr;
		info.flags = 0;
		info.bindingCount = static_cast<uint32_t>(vk_bindings.size());
		info.pBindings = vk_bindings.data();

		VkResult res = vkCreateDescriptorSetLayout(*context, &info, NULL, &_layout);
		VclCheck(res == VK_SUCCESS, "Pipeline state was created.");
	}

	DescriptorSetLayout::~DescriptorSetLayout()
	{
		vkDestroyDescriptorSetLayout(*_context, _layout, nullptr);
	}

	uint32_t DescriptorSetLayout::nrDescriptors(DescriptorType type) const
	{
		return _descriptor_summary[static_cast<size_t>(type)];
	}

	DescriptorSet::DescriptorSet
	(
		const DescriptorSetLayout* layout
	)
	: _layout(layout)
	{
		const auto& ctx = *_layout->context();

		// Create a descriptor pool that matches the layout
		std::vector<VkDescriptorPoolSize> pool_sizes;
		const auto types = { DescriptorType::Sampler, DescriptorType::CombinedImageSampler, DescriptorType::SampledImage, DescriptorType::StorageImage, DescriptorType::UniformTexelBuffer, DescriptorType::StorageTexelBuffer, DescriptorType::UniformBuffer, DescriptorType::StorageBuffer, DescriptorType::UniformBufferDynamic, DescriptorType::StorageBufferDynamic, DescriptorType::InputAttachment };
		for (const auto& type : types)
		{
			VkDescriptorPoolSize pool = {};
			pool.type = convert(type);
			pool.descriptorCount = _layout->nrDescriptors(type);
			if (pool.descriptorCount > 0)
				pool_sizes.emplace_back(pool);
		}

		VkDescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
		pool_info.pPoolSizes = pool_sizes.data();
		pool_info.maxSets = 1;
		VkResult res = vkCreateDescriptorPool(ctx, &pool_info, nullptr, &_pool);
		VclCheck(res == VK_SUCCESS, "Descriptor pool was created.");

		// Allocate a descriptor set from the previously allocated pool
		VkDescriptorSetAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		alloc_info.descriptorPool = _pool;
		alloc_info.descriptorSetCount = 1;

		VkDescriptorSetLayout l = *layout;
		alloc_info.pSetLayouts = &l;

		// Allocate 'descriptorSetCount' descriptor sets. Each corresponding to
		// one of the entries in 'pSetLayouts'.
		res = vkAllocateDescriptorSets(ctx, &alloc_info, &_set);
		VclEnsure(res == VK_SUCCESS, "Descriptor set was created.");
	}

	DescriptorSet::~DescriptorSet()
	{
		const auto& ctx = *_layout->context();
		vkFreeDescriptorSets(ctx, _pool, 1, &_set);
		vkDestroyDescriptorPool(ctx, _pool, nullptr);
	}

	bool DescriptorSet::update
	(
		stdext::span<const UpdateDescriptor> update_descriptors
	)
	{
		if (update_descriptors.empty())
			return true;

		// Build array offsets and sizes
		std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> array_infos;
		for (const auto& desc : update_descriptors)
		{
			auto& entry = array_infos.emplace(desc.Binding, std::make_pair(0, 0)).first->second;
			entry.first = std::min<uint32_t>(desc.ArrayOffset, entry.first);
			entry.second++;
		}

		// Organize the resources to update
		std::unordered_map<uint32_t, std::tuple<VkDescriptorType, uint32_t, std::vector<VkDescriptorBufferInfo>>> buffer_infos;
		for (const auto& desc : update_descriptors)
		{
			auto& buffer = absl::get<BufferView>(desc.Resource);
			VkDescriptorBufferInfo buffer_info = {};
			buffer_info.buffer = buffer.Id;
			buffer_info.offset = buffer.Offset;
			buffer_info.range = buffer.Size;

			const auto& array_info = array_infos[desc.Binding];
			const auto array_base = array_info.first;
			const auto array_size = array_info.second;
			auto& entry = buffer_infos.emplace(desc.Binding, std::make_tuple(convert(desc.Type),
			                                   array_base,
				                               std::vector<VkDescriptorBufferInfo>(array_size))).first->second;
			auto& buffers = std::get<2>(entry);
			buffers[desc.ArrayOffset - array_base] = buffer_info;
		}

		std::vector<VkWriteDescriptorSet> write_sets;
		for (const auto& resource : buffer_infos)
		{
			VkWriteDescriptorSet descriptor_write = {};
			descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptor_write.dstSet = _set;
			descriptor_write.dstBinding = resource.first;
			descriptor_write.dstArrayElement = 0;
			descriptor_write.descriptorCount = static_cast<uint32_t>(std::get<2>(resource.second).size());
			descriptor_write.descriptorType = std::get<0>(resource.second);

			descriptor_write.pBufferInfo = std::get<2>(resource.second).data();
			descriptor_write.pImageInfo = nullptr;
			descriptor_write.pTexelBufferView = nullptr;

			write_sets.emplace_back(descriptor_write);
		}

		const auto& ctx = *_layout->context();
		vkUpdateDescriptorSets(ctx, static_cast<uint32_t>(write_sets.size()), write_sets.data(), 0, nullptr);
		return true;
	}
}}}
