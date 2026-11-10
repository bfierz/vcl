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
#include <cstdint>
#include <deque>
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/runtime/state/sampler.h>

namespace Vcl { namespace Graphics { namespace Runtime {
	//! Sampler baked into the bindless layout.
	//! HLSL: register(s<Register>, space1); Vulkan: set 1, binding <Register>
	struct StaticSampler
	{
		uint32_t Register{ 0 };
		SamplerDescription Sampler;
	};

	//! Device-wide bindless layout (Vulkan / D3D12).
	//! One layout is shared by all graphics and compute pipelines of a device.
	struct BindlessLayoutDescription
	{
		//! Number of slots in the resource heap (buffers and textures)
		uint32_t ResourceCapacity{ 65536 };

		//! Number of slots in the sampler heap.
		//! D3D12 limits shader-visible sampler heaps to 2048 entries.
		uint32_t SamplerCapacity{ 2048 };

		//! Size of the push-constant / root-constant block in bytes.
		//! Must be a multiple of 4. 128 bytes are available on every Vulkan device.
		uint32_t PushConstantSize{ 128 };

		//! Samplers available to every shader without a heap index
		std::vector<StaticSampler> StaticSamplers;
	};

	//! What a slot of the resource heap is viewed as in the shader
	enum class ResourceViewType
	{
		ConstantBuffer, //!< HLSL: ConstantBuffer<T>; Vulkan: uniform buffer
		Buffer,         //!< HLSL: (ByteAddress|Structured)Buffer; Vulkan: storage buffer
		RWBuffer,       //!< HLSL: RW(ByteAddress|Structured)Buffer; Vulkan: storage buffer
		Texture,        //!< HLSL: Texture*<T>; Vulkan: sampled image
		RWTexture       //!< HLSL: RWTexture*<T>; Vulkan: storage image
	};

	//! Index into the resource heap, passed to shaders through push constants or buffers
	struct ResourceIndex
	{
		enum : uint32_t
		{
			Invalid = 0xffffffffu
		};

		uint32_t Value{ Invalid };

		bool isValid() const { return Value != Invalid; }
	};

	//! Index into the sampler heap, passed to shaders through push constants or buffers
	struct SamplerIndex
	{
		enum : uint32_t
		{
			Invalid = 0xffffffffu
		};

		uint32_t Value{ Invalid };

		bool isValid() const { return Value != Invalid; }
	};

	//! Allocator for the slots of a bindless heap.
	//! A released slot is not reused immediately, because the GPU may still read
	//! it. It is retired with the fence value of the work recorded at that time
	//! and becomes available again once that fence value is completed.
	class BindlessIndexAllocator
	{
	public:
		enum : uint32_t
		{
			InvalidIndex = 0xffffffffu
		};

		explicit BindlessIndexAllocator(uint32_t capacity)
		: _capacity(capacity)
		{
		}

		//! \returns the total number of slots
		uint32_t capacity() const { return _capacity; }

		//! \returns the number of slots in use
		uint32_t nrAllocated() const { return _nrAllocated; }

		//! \returns the number of released slots, which are not available yet
		uint32_t nrRetired() const { return static_cast<uint32_t>(_retired.size()); }

		//! Allocate a slot
		//! \returns The slot index, or InvalidIndex if no slot is available
		uint32_t allocate()
		{
			uint32_t index = InvalidIndex;
			if (!_free.empty())
			{
				index = _free.back();
				_free.pop_back();
			} else if (_nrUsed < _capacity)
			{
				index = _nrUsed++;
			} else
			{
				return InvalidIndex;
			}

			_nrAllocated++;
			return index;
		}

		//! Release a slot. It is available again after the current fence value completed.
		void retire(uint32_t index)
		{
			VclRequire(index < _nrUsed, "Index was allocated.");
			VclRequire(_nrAllocated > 0, "Slots are allocated.");

			_retired.push_back({ _fenceValue, index });
			_nrAllocated--;
		}

		//! Mark a frame boundary
		//! \param next_fence_value Fence value signalled when the work recorded from now on is finished
		//! \param completed_fence_value Most recent fence value the GPU completed
		//! \param on_free Called with the index of each slot that becomes available
		template<typename Func>
		void advance(uint64_t next_fence_value, uint64_t completed_fence_value, Func&& on_free)
		{
			VclRequire(next_fence_value >= _fenceValue, "Fence values are monotonic.");

			while (!_retired.empty() && _retired.front().FenceValue <= completed_fence_value)
			{
				const uint32_t index = _retired.front().Index;
				_retired.pop_front();

				on_free(index);
				_free.push_back(index);
			}
			_fenceValue = next_fence_value;
		}

		void advance(uint64_t next_fence_value, uint64_t completed_fence_value)
		{
			advance(next_fence_value, completed_fence_value, [](uint32_t) {});
		}

	private:
		struct RetiredSlot
		{
			uint64_t FenceValue;
			uint32_t Index;
		};

		//! Total number of slots
		uint32_t _capacity{ 0 };

		//! Number of slots handed out at least once
		uint32_t _nrUsed{ 0 };

		//! Number of slots currently in use
		uint32_t _nrAllocated{ 0 };

		//! Fence value assigned to slots retired now
		uint64_t _fenceValue{ 0 };

		//! Slots available for reuse
		std::vector<uint32_t> _free;

		//! Released slots in order of their fence values
		std::deque<RetiredSlot> _retired;
	};
}}}
