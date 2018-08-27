/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2016 Basil Fierz
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
#include <memory>

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/graphics/runtime/resource/buffer.h>
#include <vcl/graphics/vulkan/memory.h>

#ifdef VCL_VULKAN_SUPPORT

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan
{
	VCL_DECLARE_FLAGS(BufferUsage,
		
		//! The buffer can be used as the source operand of transfer operations (vkCmdCopyBuffer, vkCmdCopyBufferToImage). 
		TransferSource,
		
		//! The buffer can be used as the destination operand of transfer operations(vkCmdCopyBuffer, vkCmdCopyImageToBuffer, vkCmdUpdateBuffer, vkCmdFillBuffer, vkCmdWriteTimestamp, vkCmdCopyQueryPoolResults).
		TransferDestination,

		//! The buffer supports reads via uniform texel buffer descriptors.
		UniformTexelBuffer,

		//! The buffer supports loads, stores, and atomic operations via storage texel buffer descriptors.
		StorageTexelBuffer,

		//! The buffer supports reads via uniform buffer descriptors.
		UniformBuffer,

		//! The buffer supports loads, stores, and atomic operations via storage buffer descriptors.
		StorageBuffer,

		//! The buffer can be bound as an index buffer using the vkCmdBindIndexBuffer command.
		IndexBuffer,

		//! The buffer can be bound as a vertex buffer using the vkCmdBindVertexBuffers command.
		VertexBuffer,

		//! The buffer can be used as the source of indirect commands(vkCmdDrawIndirect, vkCmdDrawIndexedIndirect, vkCmdDispatchIndirect).
		IndirectBuffer
	);


	class Buffer final : public Runtime::Buffer
	{
	public:
		Buffer(Vcl::Graphics::Vulkan::Context* context, const BufferDescription& desc, Flags<BufferUsage> usage, const BufferInitData* init_data = nullptr, Vcl::Graphics::Vulkan::Memory* memory = nullptr);
		virtual ~Buffer();
		
	public:
		Vcl::Graphics::Vulkan::Memory* memory() const { return _memory; }

		VkBuffer id() const { return _buffer; }

	private:
		//! Buffer object
		VkBuffer _buffer{ nullptr };

		//! Memory region to which this buffer belongs
		Vcl::Graphics::Vulkan::Memory* _memory{ nullptr };

		//! Delete memory when allocated within this object
		std::unique_ptr<Vcl::Graphics::Vulkan::Memory> _memoryOwner;
	};
}}}}
#endif // VCL_VULKAN_SUPPORT
