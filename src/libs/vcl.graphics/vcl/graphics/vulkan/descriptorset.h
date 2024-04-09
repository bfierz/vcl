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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <tuple>

// Abseil
#include <absl/types/variant.h>

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/core/flags.h>
#include <vcl/core/span.h>
#include <vcl/graphics/vulkan/context.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	enum class DescriptorType
	{
		Sampler = 0,
		CombinedImageSampler = 1,
		SampledImage = 2,
		StorageImage = 3,
		UniformTexelBuffer = 4,
		StorageTexelBuffer = 5,
		UniformBuffer = 6,
		StorageBuffer = 7,
		UniformBufferDynamic = 8,
		StorageBufferDynamic = 9,
		InputAttachment = 10
	};
	VkDescriptorType convert(DescriptorType type);

	VCL_DECLARE_FLAGS(ShaderStage,
		Vertex,
		TessellationControl,
		TessellationEvaluation,
		Geometry,
		Fragment,
		Compute
	);
	VkShaderStageFlags convert(Flags<ShaderStage> stages);

	struct DescriptorSetLayoutBinding
	{
		//! Type of the resource 
		DescriptorType Type;

		//! Number of entries in the array
		uint32_t Entries;

		//! Shader stages to which the resource will be accessible
		Flags<ShaderStage> ShaderStages;

		//! Binding point
		uint32_t Binding;
	};

	//! Merges two descriptor set layouts if they are compatible
	//! \returns The merged list of bindings. Returns an empty list if the input is not compatible
	std::vector<DescriptorSetLayoutBinding> merge
	(
		stdext::span<const DescriptorSetLayoutBinding> bindings0,
		stdext::span<const DescriptorSetLayoutBinding> bindings1
	);

	class DescriptorSetLayout
	{
	public:
		//! Constructor
		DescriptorSetLayout(Context* context, stdext::span<const DescriptorSetLayoutBinding> bindings);

		//! Descrutor
		~DescriptorSetLayout();

		//! Bound context
		Context* context() const { return _context; }

		//! Convert to Vulkan ID
		inline operator VkDescriptorSetLayout() const { return _layout; }

		//! Query the number of a given descriptor type
		//! \param type Type of descriptor to query the number of entries
		//! \returns The number of descriptors for \p type
		uint32_t nrDescriptors(DescriptorType type) const;

	private:
		//! Owner
		Context* _context;

		//! Vulkan descriptor layout
		VkDescriptorSetLayout _layout;

		//! Summary of used descriptors
		std::vector<uint32_t> _descriptor_summary;
	};

	class DescriptorSet final
	{
	public:
		struct BufferView
		{
			const VkBuffer Id;

			uint32_t Offset;

			uint32_t Size;
		};

		struct UpdateDescriptor
		{
			//! Type of the attached resource
			DescriptorType Type;

			//! Binding point
			uint16_t Binding;

			//! Array offset
			uint16_t ArrayOffset;

			//! Resource to write
			absl::variant<BufferView> Resource;
		};

		//! Constructor
		DescriptorSet
		(
			const DescriptorSetLayout* layout
		);

		//! Descrutor
		~DescriptorSet();

		//! Convert to Vulkan ID
		inline operator VkDescriptorSet() const
		{
			return _set;
		}

		inline const VkDescriptorSet* ptr() const
		{
			return &_set;
		}

		//! Update the resources bound to the descriptor set
		bool update
		(
			stdext::span<const UpdateDescriptor> buffers
		);

	private:
		//! Pool to allocate descriptors from
		VkDescriptorPool _pool;

		//! Layout
		const DescriptorSetLayout* _layout;

		//! Descriptor set
		VkDescriptorSet _set;
	};

	struct PushConstantDescriptor
	{
		using ShaderStages = Flags<Graphics::Vulkan::ShaderStage>;

		ShaderStages StageFlags;
		uint32_t     Offset;
		uint32_t     Size;
	};
}}}
