/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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
#include <array>
#include <vector>

// Abseil
#include <absl/types/variant.h>

// Windows Runtime Library
#include <wrl.h>

// VCL
#include <vcl/core/flags.h>
#include <vcl/core/span.h>
#include <vcl/graphics/d3d12/semaphore.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>

namespace Vcl { namespace Graphics { namespace D3D12
{
	class Device;

	enum class DescriptorTableLayoutEntryType
	{
		Table = 0,
		Constant = 1,
		InlineConstantBufferView = 2,
		InlineShaderResourceView = 3,
		InlineUnorderedAccessView = 4
	};

	struct TableDescriptorEntry
	{
		D3D12_DESCRIPTOR_RANGE_TYPE RangeType;
		uint32_t NumDescriptors;
		uint32_t BaseShaderRegister;
		uint32_t RegisterSpace;
		D3D12_DESCRIPTOR_RANGE_FLAGS Flags;
		uint32_t OffsetInDescriptorsFromTableStart;
	};

	struct TableDescriptor
	{
		std::vector<TableDescriptorEntry> Entries;
	};

	struct ContantDescriptor
	{
		uint32_t ShaderRegister;
		uint32_t RegisterSpace;
		uint32_t Num32BitValues;
	};

	struct InlineDescriptor
	{
		uint32_t ShaderRegister;
		uint32_t RegisterSpace;
		D3D12_ROOT_DESCRIPTOR_FLAGS Flags;
	};

	struct DescriptorTableLayoutEntry
	{
		//! Type of the represented descriptor
		DescriptorTableLayoutEntryType Type;

		//! Descriptor
		absl::variant<TableDescriptor, ContantDescriptor, InlineDescriptor> Descriptor;

		//! Shader stage visibility
		D3D12_SHADER_VISIBILITY Visibility;
	};

	class DescriptorTableLayout
	{
	public:
		template<typename T>
		using ComPtr = Microsoft::WRL::ComPtr<T>;

		DescriptorTableLayout
		(
			Device* device,
			std::vector<DescriptorTableLayoutEntry> entries = {},
			std::vector<D3D12_STATIC_SAMPLER_DESC> static_samplers = {},
			bool use_input_assembler = true
		);

		ID3D12RootSignature* rootSignature() const { return _d3d12RootSignature.Get(); }

		int numberOfRangeDescriptors() const { return static_cast<int>(_rangeDescriptors.size()); }

		stdext::span<const D3D12_DESCRIPTOR_RANGE1> rangeDescriptors() const { return _rangeDescriptors; }

	private:
		ComPtr<ID3D12RootSignature> _d3d12RootSignature;

		//! Decriptor table range descriptors
		std::vector<D3D12_DESCRIPTOR_RANGE1> _rangeDescriptors;

		//! Dynamic resource descriptors
		std::vector<DescriptorTableLayoutEntry> _dynamicResources;

		//! Static sampler descriptors
		std::vector<D3D12_STATIC_SAMPLER_DESC> _staticSamplers;
	};

	class DescriptorTable
	{
	public:
		template<typename T>
		using ComPtr = Microsoft::WRL::ComPtr<T>;

		DescriptorTable(Device* device, DescriptorTableLayout* layout);

		DescriptorTableLayout* layout() { return _layout; }
		void setToCompute(ID3D12GraphicsCommandList* cmd_list);
		void setToGraphics(ID3D12GraphicsCommandList* cmd_list);

		void addResource(int descriptor_index, Runtime::D3D12::Buffer* buffer, uint64_t first, uint32_t count, uint32_t stride);

	private:
		//! Associated device
		Device* _device;

		//! Native descriptor heap object
		ComPtr<ID3D12DescriptorHeap> _d3d12DescriptorHeap;

		//! Heap layout
		DescriptorTableLayout* _layout;

		//! Current offset into descriptor heap
		INT _offset{ 0 };

		//! Bound resources
		std::vector<Runtime::D3D12::Resource*> _resources;

		//! Descriptor heap increments
		std::array<UINT, D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES> _heapIncrements;
	};
}}}
