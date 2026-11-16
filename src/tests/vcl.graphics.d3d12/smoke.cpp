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

// Checks what the platform's D3D12 runtime provides. Written to confirm the
// D3D12 runtime on WSL: DXCore adapter enumeration, the debug layer, WARP and
// shader model 6.6 bindless access. Each check prints what it found.

// C++ standard library
#include <chrono>
#include <iostream>
#include <vector>

// VCL
#include <vcl/graphics/d3d12/d3d.h>
#include <directx/d3dx12.h>
#include <vcl/graphics/d3d12/device.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>

// Google test
#include <gtest/gtest.h>

#include "bindless.cs.hlsl.cso.h"

extern std::unique_ptr<Vcl::Graphics::D3D12::Device> device;

namespace {
	using Microsoft::WRL::ComPtr;

	const char* toString(D3D_SHADER_MODEL model)
	{
		switch (model)
		{
		case D3D_SHADER_MODEL_5_1: return "5.1";
		case D3D_SHADER_MODEL_6_0: return "6.0";
		case D3D_SHADER_MODEL_6_1: return "6.1";
		case D3D_SHADER_MODEL_6_2: return "6.2";
		case D3D_SHADER_MODEL_6_3: return "6.3";
		case D3D_SHADER_MODEL_6_4: return "6.4";
		case D3D_SHADER_MODEL_6_5: return "6.5";
		case D3D_SHADER_MODEL_6_6: return "6.6";
		case D3D_SHADER_MODEL_6_7: return "6.7";
		case D3D_SHADER_MODEL_6_8: return "6.8";
		case D3D_SHADER_MODEL_6_9: return "6.9";
		default: return "unknown";
		}
	}

	//! \returns the highest shader model supported by 'device'
	D3D_SHADER_MODEL highestShaderModel(ID3D12Device* device)
	{
		// The runtime rejects models it doesn't know, so query from the newest down
		const D3D_SHADER_MODEL models[] = {
			D3D_SHADER_MODEL_6_9, D3D_SHADER_MODEL_6_8, D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_6,
			D3D_SHADER_MODEL_6_5, D3D_SHADER_MODEL_6_4, D3D_SHADER_MODEL_6_3, D3D_SHADER_MODEL_6_2,
			D3D_SHADER_MODEL_6_1, D3D_SHADER_MODEL_6_0
		};
		for (auto model : models)
		{
			D3D12_FEATURE_DATA_SHADER_MODEL data = { model };
			if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &data, sizeof(data))))
				return data.HighestShaderModel;
		}
		return D3D_SHADER_MODEL_5_1;
	}

	D3D12_RESOURCE_BINDING_TIER resourceBindingTier(ID3D12Device* device)
	{
		D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
		device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
		return options.ResourceBindingTier;
	}
}

TEST(D3D12Smoke, EnumeratesAdapters)
{
	using namespace Vcl::Graphics::D3D12;

	const auto adapters = enumerateAdapters();
	bool has_software_adapter = false;
	for (const auto& adapter : adapters)
	{
		std::cout << "Adapter: " << adapter.name << (adapter.isHardware ? " (hardware, " : " (software, ")
				  << (adapter.dedicatedVideoMemory >> 20) << " MiB dedicated)" << std::endl;
		has_software_adapter = has_software_adapter || !adapter.isHardware;
	}
	std::cout << "WARP adapter: " << (has_software_adapter ? "yes" : "no") << std::endl;
	RecordProperty("warp", has_software_adapter ? "yes" : "no");

	ASSERT_FALSE(adapters.empty());
	EXPECT_GE(selectAdapter(adapters, device->adapter().isHardware ? DeviceType::Hardware : DeviceType::Warp), 0);
}

TEST(D3D12Smoke, ReportsCapabilities)
{
	ComPtr<ID3D12Debug> debug;
	const bool has_debug_layer = SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
	const auto shader_model = highestShaderModel(device->nativeDevice());
	const auto binding_tier = resourceBindingTier(device->nativeDevice());

	std::cout << "Debug layer: " << (has_debug_layer ? "yes" : "no") << std::endl;
	std::cout << "Highest shader model: " << toString(shader_model) << std::endl;
	std::cout << "Resource binding tier: " << binding_tier << std::endl;
	RecordProperty("debug_layer", has_debug_layer ? "yes" : "no");
	RecordProperty("shader_model", toString(shader_model));
	RecordProperty("resource_binding_tier", binding_tier);

	EXPECT_GE(shader_model, D3D_SHADER_MODEL_6_0);
}

TEST(D3D12Smoke, BindlessCompute)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;

	auto native_device = device->nativeDevice();
	const auto shader_model = highestShaderModel(native_device);
	if (shader_model < D3D_SHADER_MODEL_6_6 || resourceBindingTier(native_device) < D3D12_RESOURCE_BINDING_TIER_3)
		GTEST_SKIP() << "Bindless access needs shader model 6.6 and resource binding tier 3 (found "
					 << toString(shader_model) << ", tier " << resourceBindingTier(native_device) << ")";

	// Root signature: only the descriptor indices; the shader indexes the heap itself
	CD3DX12_ROOT_PARAMETER1 indices_param;
	indices_param.InitAsConstants(3, 0);
	CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC signature_desc;
	signature_desc.Init_1_1(1, &indices_param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);
	auto signature = device->createRootSignature(signature_desc);
	ASSERT_TRUE(signature);

	D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
	pso_desc.pRootSignature = signature.Get();
	pso_desc.CS = { BindlessCsoCS.data(), BindlessCsoCS.size() };
	auto pso = device->createComputePipelineState(pso_desc);
	ASSERT_TRUE(pso);

	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);

	constexpr uint32_t count = 256;
	std::vector<float> input_values(count);
	for (uint32_t i = 0; i < count; i++)
		input_values[i] = static_cast<float>(i);

	BufferInitData input_data = { input_values.data(), count * sizeof(float) };
	D3D12::Buffer input(device.get(), { count * sizeof(float), BufferUsage::Storage }, &input_data, cmd_list.Get());
	D3D12::Buffer output(device.get(), { count * sizeof(float), BufferUsage::Storage });
	D3D12::Buffer read_back(device.get(), { count * sizeof(float), BufferUsage::MapRead | BufferUsage::CopyDst });

	// Place the views at arbitrary heap indices, so that a shader ignoring the
	// indices reads the wrong descriptors
	constexpr uint32_t input_index = 5;
	constexpr uint32_t output_index = 2;
	auto heap = device->createDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8, true);
	const auto increment = native_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
	srv_desc.Format = DXGI_FORMAT_UNKNOWN;
	srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv_desc.Buffer.NumElements = count;
	srv_desc.Buffer.StructureByteStride = sizeof(float);
	native_device->CreateShaderResourceView(input.handle(), &srv_desc, CD3DX12_CPU_DESCRIPTOR_HANDLE(cpuHeapStart(heap.Get()), input_index, increment));

	D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
	uav_desc.Format = DXGI_FORMAT_UNKNOWN;
	uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uav_desc.Buffer.NumElements = count;
	uav_desc.Buffer.StructureByteStride = sizeof(float);
	native_device->CreateUnorderedAccessView(output.handle(), nullptr, &uav_desc, CD3DX12_CPU_DESCRIPTOR_HANDLE(cpuHeapStart(heap.Get()), output_index, increment));

	input.transition(cmd_list.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	output.transition(cmd_list.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	// Directly indexed heaps must be set before the root signature
	ID3D12DescriptorHeap* heaps[] = { heap.Get() };
	cmd_list->SetDescriptorHeaps(1, heaps);
	cmd_list->SetComputeRootSignature(signature.Get());
	cmd_list->SetPipelineState(pso.Get());

	const uint32_t indices[] = { input_index, output_index, count };
	cmd_list->SetComputeRoot32BitConstants(0, 3, indices, 0);
	cmd_list->Dispatch(count / 64, 1, 1);

	output.transition(cmd_list.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
	cmd_list->CopyResource(read_back.handle(), output.handle());
	VCL_DIRECT3D_SAFE_CALL(cmd_list->Close());

	ID3D12CommandList* const lists[] = { cmd_list.Get() };
	device->defaultQueue()->nativeQueue()->ExecuteCommandLists(1, lists);
	ASSERT_TRUE(device->defaultQueue()->sync(std::chrono::seconds{ 10 }));

	const auto result = static_cast<const float*>(read_back.map({ 0, count * sizeof(float) }));
	std::vector<float> output_values(result, result + count);
	read_back.unmap();

	for (uint32_t i = 0; i < count; i++)
		EXPECT_EQ(2.0f * i + 1.0f, output_values[i]) << "at index " << i;
}
