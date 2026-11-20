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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <array>
#include <vector>

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/graphics/runtime/resource/texture.h>
#include <vcl/graphics/runtime/state/resourcetable.h>
#include <vcl/graphics/runtime/vulkan/resource/buffer.h>
#include <vcl/graphics/runtime/vulkan/resource/texture.h>
#include <vcl/graphics/vulkan/context.h>

namespace Vcl { namespace Graphics { namespace Vulkan {
	class CommandBuffer;
	class DescriptorHeap;

	using ResourceView = Runtime::BindlessView<DescriptorHeap, Runtime::ResourceIndex>;
	using SamplerView = Runtime::BindlessView<DescriptorHeap, Runtime::SamplerIndex>;

	/*!
	 * \brief Bindless descriptor heap and the device-wide pipeline layout
	 *
	 * Every pipeline of a context uses the pipeline layout of the heap:
	 * - Set 0, binding 0: Resource heap. A partially bound array of mutable
	 *   descriptors (uniform buffer, storage buffer, sampled image, storage image).
	 * - Set 0, binding 1: Sampler heap
	 * - Set 1, binding N: Static sampler with register N
	 * - One push-constant range visible to all stages
	 *
	 * DXC maps SM 6.6 heap indexing to this layout with
	 * `-fvk-bind-resource-heap 0 0 -fvk-bind-sampler-heap 1 0`.
	 *
	 * Buffers and textures don't know the heap. Views are created explicitly
	 * and must not outlive the resource they refer to while shaders access them.
	 */
	class DescriptorHeap final
	{
	public:
		//! Set and bindings of the layout
		enum : uint32_t
		{
			HeapSet = 0,
			ResourceHeapBinding = 0,
			SamplerHeapBinding = 1,
			StaticSamplerSet = 1
		};

		DescriptorHeap(Context* context, const Runtime::BindlessLayoutDescription& desc);
		DescriptorHeap(const DescriptorHeap&) = delete;
		DescriptorHeap(DescriptorHeap&&) = delete;

		//! Destructor. Requires all views to be released.
		~DescriptorHeap();

		DescriptorHeap& operator=(const DescriptorHeap&) = delete;
		DescriptorHeap& operator=(DescriptorHeap&&) = delete;

		//! \returns the description the heap was created with
		const Runtime::BindlessLayoutDescription& desc() const { return _desc; }

		//! \returns the pipeline layout shared by all pipelines
		VkPipelineLayout layout() const { return _pipelineLayout; }

		//! \returns the number of resource views in use
		uint32_t nrResourceViews() const { return _resources.nrAllocated(); }

		//! \returns the number of samplers in use
		uint32_t nrSamplers() const { return _samplers.nrAllocated(); }

		//! Create a view on a range of a buffer
		//! \param buffer Buffer to view. Requires the uniform usage for
		//!        constant buffers and the storage usage otherwise.
		//! \param type ConstantBuffer, Buffer or RWBuffer
		//! \param offset Offset of the range in bytes
		//! \param size Size of the range in bytes
		//! \param stride Size of an element of a structured buffer; 0 for a raw
		//!        buffer. Vulkan takes the element layout from the shader.
		ResourceView createView(const Runtime::Vulkan::Buffer& buffer, Runtime::ResourceViewType type, uint64_t offset, uint64_t size, uint32_t stride = 0);

		//! Create a view on a texture
		//! \param texture Texture to view. Requires the sampled usage for
		//!        'Texture' and the storage usage for 'RWTexture'.
		//! \param type Texture or RWTexture
		//! \param view Part of the texture to view. RWTexture views cover one mip-map level.
		//! \note Shaders expect sampled textures in the shader-read-only layout
		//!       and read-write textures in the general layout.
		ResourceView createView(const Runtime::Vulkan::Texture& texture, Runtime::ResourceViewType type, const Runtime::TextureViewDescription& view = {});

		//! Create a sampler in the sampler heap
		SamplerView createSampler(const Runtime::SamplerDescription& desc);

		//! \returns the type of the view stored in a slot
		Runtime::ResourceViewType viewType(Runtime::ResourceIndex index) const;

		//! Mark a frame boundary. Views released from now on are tagged with
		//! \p next_fence_value. Slots tagged with a value less or equal to
		//! \p completed_fence_value become available again.
		void advance(uint64_t next_fence_value, uint64_t completed_fence_value);

		//! Bind the heaps for graphics and compute pipelines.
		//! Required once per command buffer.
		void bind(CommandBuffer& cmd_buffer) const;

	private:
		template<typename, typename>
		friend class Runtime::BindlessView;

		void retire(Runtime::ResourceIndex index);
		void retire(Runtime::SamplerIndex index);

		//! Allocate a slot of the resource heap
		Runtime::ResourceIndex allocateResource(Runtime::ResourceViewType type);

		//! Owner
		Context* _context{ nullptr };

		//! Description of the heap
		Runtime::BindlessLayoutDescription _desc;

		//! Layouts of the heap set and the static sampler set
		std::array<VkDescriptorSetLayout, 2> _setLayouts{ VK_NULL_HANDLE, VK_NULL_HANDLE };

		//! Pipeline layout shared by all pipelines
		VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };

		//! Pool the sets are allocated from
		VkDescriptorPool _pool{ VK_NULL_HANDLE };

		//! Heap set and static sampler set
		std::array<VkDescriptorSet, 2> _sets{ VK_NULL_HANDLE, VK_NULL_HANDLE };

		//! Number of sets in use (1 without static samplers)
		uint32_t _nrSets{ 1 };

		//! Samplers baked into the layout
		std::vector<VkSampler> _staticSamplers;

		//! Slots of the resource heap
		Runtime::BindlessIndexAllocator _resources;

		//! Slots of the sampler heap
		Runtime::BindlessIndexAllocator _samplers;

		//! View type of each resource slot
		std::vector<Runtime::ResourceViewType> _viewTypes;

		//! Image view owned by each resource slot (texture views only)
		std::vector<VkImageView> _imageViews;

		//! Sampler owned by each sampler slot
		std::vector<VkSampler> _samplerObjects;
	};
}}}
