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
#include <vcl/graphics/d3d12/descriptortable.h>

// C++ standard library
#include <numeric>
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/d3d12/device.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	DescriptorTableLayout::DescriptorTableLayout(
		Device* device,
		std::vector<DescriptorTableLayoutEntry> entries,
		std::vector<D3D12_STATIC_SAMPLER_DESC> static_samplers,
		bool use_input_assembler)
	: _dynamicResources{ std::move(entries) }
	, _staticSamplers{ std::move(static_samplers) }
	{
		auto& descriptor_ranges = _rangeDescriptors;
		int num_range_descr = std::accumulate(std::begin(_dynamicResources), std::end(_dynamicResources), 0,
			[](int sum, const DescriptorTableLayoutEntry& entry) -> int
			{
				if (entry.Type == DescriptorTableLayoutEntryType::Table)
				{
					return sum + static_cast<int>(absl::get<TableDescriptor>(entry.Descriptor).Entries.size());
				}
				
				return sum;
			});
		descriptor_ranges.reserve(num_range_descr);

		std::vector<CD3DX12_ROOT_PARAMETER1> root_params;
		root_params.reserve(_dynamicResources.size());
		std::transform(std::begin(_dynamicResources), std::end(_dynamicResources), std::back_inserter(root_params),
			[&descriptor_ranges](const DescriptorTableLayoutEntry& entry)
			{
				CD3DX12_ROOT_PARAMETER1 param;
				if (entry.Type == DescriptorTableLayoutEntryType::Constant)
				{
					const auto& descr = absl::get<ContantDescriptor>(entry.Descriptor);
					param.InitAsConstants(descr.Num32BitValues, descr.ShaderRegister, descr.RegisterSpace, entry.Visibility);
				}
				else if (entry.Type == DescriptorTableLayoutEntryType::Table)
				{
					const auto& descr = absl::get<TableDescriptor>(entry.Descriptor);
					const auto first_entry = descriptor_ranges.data() + descriptor_ranges.size();

					std::transform(std::begin(descr.Entries), std::end(descr.Entries), std::back_inserter(descriptor_ranges),
						[](const TableDescriptorEntry& entry)
						{
							D3D12_DESCRIPTOR_RANGE1 range;
							range.RangeType = entry.RangeType;
							range.NumDescriptors = entry.NumDescriptors;
							range.BaseShaderRegister = entry.BaseShaderRegister;
							range.RegisterSpace = entry.RegisterSpace;
							range.Flags = entry.Flags;
							range.OffsetInDescriptorsFromTableStart = entry.OffsetInDescriptorsFromTableStart;
							return range;
						});

					param.InitAsDescriptorTable(descr.Entries.size(), first_entry, entry.Visibility);
				}
				else if (entry.Type == DescriptorTableLayoutEntryType::InlineConstantBufferView)
				{
					const auto& descr = absl::get<InlineDescriptor>(entry.Descriptor);
					param.InitAsConstantBufferView(descr.ShaderRegister, descr.RegisterSpace, descr.Flags, entry.Visibility);
				}
				else if (entry.Type == DescriptorTableLayoutEntryType::InlineShaderResourceView)
				{
					const auto& descr = absl::get<InlineDescriptor>(entry.Descriptor);
					param.InitAsShaderResourceView(descr.ShaderRegister, descr.RegisterSpace, descr.Flags, entry.Visibility);
				}
				else if (entry.Type == DescriptorTableLayoutEntryType::InlineUnorderedAccessView)
				{
					const auto& descr = absl::get<InlineDescriptor>(entry.Descriptor);
					param.InitAsUnorderedAccessView(descr.ShaderRegister, descr.RegisterSpace, descr.Flags, entry.Visibility);
				}
				else
				{
					VclCheck(false, "Inline descriptors are not implented.");
				}

				return param;
			});


		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC sig_desc = {};
		sig_desc.Init_1_1(
			root_params.size(), root_params.data(),
			_staticSamplers.size(), _staticSamplers.data(),
			use_input_assembler ? D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT : D3D12_ROOT_SIGNATURE_FLAG_NONE);
		_d3d12RootSignature = device->createRootSignature(sig_desc);
	}

	DescriptorTable::DescriptorTable(Device* device, DescriptorTableLayout* layout)
	: _device{ device }
	, _layout{ layout }
	{
		auto d3d12_device = device->nativeDevice();
		_d3d12DescriptorHeap = device->createDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, _layout->numberOfRangeDescriptors(), true);
		for (size_t i = 0; i < _heapIncrements.size(); i++)
		{
			const auto type = static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(i);
			_heapIncrements[i] = d3d12_device->GetDescriptorHandleIncrementSize(type);
		}

		_resources.resize(_layout->numberOfRangeDescriptors());
	}

	void DescriptorTable::setToCompute(ID3D12GraphicsCommandList* cmd_list, uint32_t root_index)
	{
		// Transition the resources that are bound here
		for (size_t i = 0; i < _resources.size(); ++i)
		{
			auto resource = _resources[i];
			const auto type = _layout->rangeDescriptors()[i].RangeType;
			D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
			if (type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV)
			{
				VclCheck(resource->resourcesStates() & D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "Resource requires storage binding setting");
				state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
			}
			else if (type == D3D12_DESCRIPTOR_RANGE_TYPE_UAV)
			{
				VclCheck(resource->resourcesStates() & D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "Resource requires storage binding setting");
				state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
			}
			else
				VclDebugError("State not implemented");

			resource->transition(cmd_list, state);
		}

		cmd_list->SetComputeRootSignature(_layout->rootSignature());

		ID3D12DescriptorHeap* ppHeaps[] = { _d3d12DescriptorHeap.Get() };
		cmd_list->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

		const auto increment = _heapIncrements[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV];
		CD3DX12_GPU_DESCRIPTOR_HANDLE handle(_d3d12DescriptorHeap->GetGPUDescriptorHandleForHeapStart(), _offset, increment);
		cmd_list->SetComputeRootDescriptorTable(root_index, handle);
	}

	void DescriptorTable::setToGraphics(ID3D12GraphicsCommandList* cmd_list, uint32_t root_index)
	{
		// Transition the resources that are bound here
		for (size_t i = 0; i < _resources.size(); ++i)
		{
			auto resource = _resources[i];
			const auto type = _layout->rangeDescriptors()[i].RangeType;
			D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
			if (resource->heapType() == D3D12_HEAP_TYPE_UPLOAD)
			{
				VclCheck(resource->resourcesStates() & D3D12_RESOURCE_STATE_GENERIC_READ, "Resource requires generic read binding setting");
				state = D3D12_RESOURCE_STATE_GENERIC_READ;
			}
			else if (type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV)
			{
				VclCheck(resource->resourcesStates() & D3D12_RESOURCE_STATE_UNORDERED_ACCESS| D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, "Resource requires storage binding setting");
				state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
			}
			else if (type == D3D12_DESCRIPTOR_RANGE_TYPE_UAV)
			{
				VclCheck(resource->resourcesStates() & D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "Resource requires storage binding setting");
				state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
			}
			else if (type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV)
			{
				VclCheck(resource->resourcesStates() & D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, "Resource requires constant buffer binding setting");
				state = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
			}
			else
				VclDebugError("State not implemented");

			resource->transition(cmd_list, state);
		}

		cmd_list->SetGraphicsRootSignature(_layout->rootSignature());

		ID3D12DescriptorHeap* ppHeaps[] = { _d3d12DescriptorHeap.Get() };
		cmd_list->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

		const auto increment = _heapIncrements[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV];
		CD3DX12_GPU_DESCRIPTOR_HANDLE handle(_d3d12DescriptorHeap->GetGPUDescriptorHandleForHeapStart(), _offset, increment);
		cmd_list->SetGraphicsRootDescriptorTable(root_index, handle);
	}

	void DescriptorTable::addResource(int descriptor_index, Runtime::D3D12::Buffer* buffer, uint64_t first, uint32_t count, uint32_t stride)
	{
		auto d3d12_device = _device->nativeDevice();
		const auto increment = _heapIncrements[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV];

		_resources[descriptor_index] = buffer;

		const auto type = _layout->rangeDescriptors()[descriptor_index].RangeType;
		if (type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV)
		{
			D3D12_CONSTANT_BUFFER_VIEW_DESC desc = {};
			desc.BufferLocation = buffer->handle()->GetGPUVirtualAddress() + first*stride;
			desc.SizeInBytes = count*stride;
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				_d3d12DescriptorHeap->GetCPUDescriptorHandleForHeapStart(), _offset + descriptor_index, increment);
			d3d12_device->CreateConstantBufferView(&desc, handle);
		}
		else if (type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV)
		{
			D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
			desc.Format = DXGI_FORMAT_UNKNOWN;
			desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
			desc.Buffer.FirstElement = first;
			desc.Buffer.NumElements = count;
			desc.Buffer.StructureByteStride = stride;
			desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				_d3d12DescriptorHeap->GetCPUDescriptorHandleForHeapStart(), _offset + descriptor_index, increment);
			d3d12_device->CreateShaderResourceView(buffer->handle(), &desc, handle);
		}
		else if (type == D3D12_DESCRIPTOR_RANGE_TYPE_UAV)
		{
			D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
			desc.Format = DXGI_FORMAT_UNKNOWN;
			desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			desc.Buffer.FirstElement = first;
			desc.Buffer.NumElements = count;
			desc.Buffer.StructureByteStride = stride;
			desc.Buffer.CounterOffsetInBytes = 0;
			desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				_d3d12DescriptorHeap->GetCPUDescriptorHandleForHeapStart(), _offset + descriptor_index, increment);
			d3d12_device->CreateUnorderedAccessView(buffer->handle(), nullptr, &desc, handle);
		}
	}

	void DescriptorTable::addResource(int descriptor_index, Runtime::D3D12::Texture* texture)
	{
		auto d3d12_device = _device->nativeDevice();
		const auto increment = _heapIncrements[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV];

		_resources[descriptor_index] = texture;
		const auto type = _layout->rangeDescriptors()[descriptor_index].RangeType;
		if (type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV)
		{
			VclDebugError("Binding textures to CBV is not supported");
		}
		else if (type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV)
		{
			const auto desc = texture->srv();
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				_d3d12DescriptorHeap->GetCPUDescriptorHandleForHeapStart(), _offset + descriptor_index, increment);
			d3d12_device->CreateShaderResourceView(texture->handle(), &desc, handle);
		}
		else if (type == D3D12_DESCRIPTOR_RANGE_TYPE_UAV)
		{
			const auto desc = texture->uav();
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				_d3d12DescriptorHeap->GetCPUDescriptorHandleForHeapStart(), _offset + descriptor_index, increment);
			d3d12_device->CreateUnorderedAccessView(texture->handle(), nullptr, &desc, handle);
		}
	}
}}}
