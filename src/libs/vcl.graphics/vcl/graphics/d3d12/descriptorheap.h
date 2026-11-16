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
#include <vcl/config/direct3d12.h>

// C++ standard library
#include <vector>

// Windows Runtime Library
#include <wrl/client.h>

// VCL
#include <vcl/graphics/d3d12/device.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/d3d12/resource/texture.h>
#include <vcl/graphics/runtime/resource/texture.h>
#include <vcl/graphics/runtime/state/resourcetable.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	class DescriptorHeap;

	using ResourceView = Runtime::BindlessView<DescriptorHeap, Runtime::ResourceIndex>;
	using SamplerView = Runtime::BindlessView<DescriptorHeap, Runtime::SamplerIndex>;

	/*!
	 * \brief Bindless descriptor heaps and the device-wide root signature
	 *
	 * Every pipeline of a device uses the root signature of the heap:
	 * - Root parameter 0: Root constants at register(b0, space0), visible to
	 *   all stages (the push constants of the bindless layout)
	 * - Shader-visible CBV/SRV/UAV heap, accessed through 'ResourceDescriptorHeap'
	 * - Shader-visible sampler heap, accessed through 'SamplerDescriptorHeap'
	 * - Static samplers at register(s<Register>, space1)
	 *
	 * Shaders index the heaps directly, which needs shader model 6.6 and
	 * resource binding tier 3.
	 *
	 * Buffers and textures don't know the heap. Views are created explicitly
	 * and must not outlive the resource they refer to while shaders access them.
	 * The heap doesn't track resource states: resources must be in a state
	 * matching the view when shaders access them.
	 */
	class DescriptorHeap final
	{
	public:
		template<typename T>
		using ComPtr = Microsoft::WRL::ComPtr<T>;

		//! Root parameters of the root signature
		enum : uint32_t
		{
			RootConstantsParameter = 0
		};

		DescriptorHeap(Device* device, const Runtime::BindlessLayoutDescription& desc);
		DescriptorHeap(const DescriptorHeap&) = delete;
		DescriptorHeap(DescriptorHeap&&) = delete;

		//! Destructor. Requires all views to be released.
		~DescriptorHeap();

		DescriptorHeap& operator=(const DescriptorHeap&) = delete;
		DescriptorHeap& operator=(DescriptorHeap&&) = delete;

		//! \returns the description the heap was created with
		const Runtime::BindlessLayoutDescription& desc() const { return _desc; }

		//! \returns the root signature shared by all pipelines
		ID3D12RootSignature* rootSignature() const { return _rootSignature.Get(); }

		//! \returns the shader-visible CBV/SRV/UAV heap
		ID3D12DescriptorHeap* resourceHeap() const { return _resourceHeap.Get(); }

		//! \returns the shader-visible sampler heap
		ID3D12DescriptorHeap* samplerHeap() const { return _samplerHeap.Get(); }

		//! \returns the number of resource views in use
		uint32_t nrResourceViews() const { return _resources.nrAllocated(); }

		//! \returns the number of samplers in use
		uint32_t nrSamplers() const { return _samplers.nrAllocated(); }

		//! Create a view on a range of a buffer
		//! \param buffer Buffer to view. Requires the uniform usage for
		//!        constant buffers and the storage usage otherwise.
		//! \param type ConstantBuffer, Buffer or RWBuffer
		//! \param offset Offset of the range in bytes. Constant buffers start at
		//!        multiples of 256 bytes.
		//! \param size Size of the range in bytes. Constant buffer views are
		//!        rounded up to multiples of 256 bytes, which must be inside the buffer.
		//! \param stride Size of an element of a structured buffer; 0 for a raw
		//!        (ByteAddressBuffer) view, whose offset and size are multiples of 4.
		ResourceView createView(const Runtime::D3D12::Buffer& buffer, Runtime::ResourceViewType type, uint64_t offset, uint64_t size, uint32_t stride = 0);

		//! Create a view on a texture
		//! \param texture Texture to view. Requires the sampled usage for
		//!        'Texture' and the storage usage for 'RWTexture'.
		//! \param type Texture or RWTexture
		//! \param view Part of the texture to view. RWTexture views cover one mip-map level.
		//! \note Shaders expect sampled textures in a shader-resource state and
		//!       read-write textures in the unordered-access state.
		ResourceView createView(const Runtime::D3D12::Texture& texture, Runtime::ResourceViewType type, const Runtime::TextureViewDescription& view = {});

		//! Create a sampler in the sampler heap
		SamplerView createSampler(const Runtime::SamplerDescription& desc);

		//! \returns the type of the view stored in a slot
		Runtime::ResourceViewType viewType(Runtime::ResourceIndex index) const;

		//! Mark a frame boundary. Views released from now on are tagged with
		//! \p next_fence_value. Slots tagged with a value less or equal to
		//! \p completed_fence_value become available again.
		void advance(uint64_t next_fence_value, uint64_t completed_fence_value);

		//! Set the heaps and the root signature for graphics and compute pipelines.
		//! Required once per command list, before the root constants are set.
		void bind(ID3D12GraphicsCommandList* cmd_list) const;

	private:
		template<typename, typename>
		friend class Runtime::BindlessView;

		void retire(Runtime::ResourceIndex index);
		void retire(Runtime::SamplerIndex index);

		//! Allocate a slot of the resource heap
		Runtime::ResourceIndex allocateResource(Runtime::ResourceViewType type);

		//! \returns the CPU handle of a slot of the resource heap
		D3D12_CPU_DESCRIPTOR_HANDLE resourceHandle(Runtime::ResourceIndex index) const;

		//! Owner
		Device* _device{ nullptr };

		//! Description of the heap
		Runtime::BindlessLayoutDescription _desc;

		//! Root signature shared by all pipelines
		ComPtr<ID3D12RootSignature> _rootSignature;

		//! Shader-visible CBV/SRV/UAV heap
		ComPtr<ID3D12DescriptorHeap> _resourceHeap;

		//! Shader-visible sampler heap
		ComPtr<ID3D12DescriptorHeap> _samplerHeap;

		//! CPU handles of the first slots
		D3D12_CPU_DESCRIPTOR_HANDLE _resourceHeapStart{};
		D3D12_CPU_DESCRIPTOR_HANDLE _samplerHeapStart{};

		//! Distance between two slots in bytes
		uint32_t _resourceIncrement{ 0 };
		uint32_t _samplerIncrement{ 0 };

		//! Slots of the resource heap
		Runtime::BindlessIndexAllocator _resources;

		//! Slots of the sampler heap
		Runtime::BindlessIndexAllocator _samplers;

		//! View type of each resource slot
		std::vector<Runtime::ResourceViewType> _viewTypes;
	};
}}}
