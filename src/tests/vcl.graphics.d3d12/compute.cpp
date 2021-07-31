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

// VCL configuration
#include <vcl/config/global.h>

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/d3d12/semaphore.h>
#include <vcl/graphics/d3d12/descriptortable.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/d3d12/resource/shader.h>

// Google test
#include <gtest/gtest.h>

#include "saxpy.cs.hlsl.cso.h"

extern std::unique_ptr<Vcl::Graphics::D3D12::Device> device;

struct SaxpyKernelParameters
{
	// Total width
	uint32_t Width;

	// Total height
	uint32_t Height;

	// Saxpy scale 'a'
	float a;
};

TEST(D3D12Compute, Saxpy)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;

	D3D12::Shader cs(ShaderType::ComputeShader, 0, SaxpyCsoCS);
	
	std::vector<DescriptorTableLayoutEntry> dynamic_resources =
	{
		{ DescriptorTableLayoutEntryType::Constant, ContantDescriptor{0, 0, 3}, D3D12_SHADER_VISIBILITY_ALL },
		{ DescriptorTableLayoutEntryType::Table, TableDescriptor{{
			{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
			{ D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND}
			}}, D3D12_SHADER_VISIBILITY_ALL }
	};

	DescriptorTableLayout table_layout{ device.get(), std::move(dynamic_resources), {} };
	auto signature = table_layout.rootSignature();

	D3D12_COMPUTE_PIPELINE_STATE_DESC ps_desc = {};
	ps_desc.pRootSignature = signature;
	ps_desc.CS.BytecodeLength = cs.data().size();
	ps_desc.CS.pShaderBytecode = cs.data().data();
	auto pso = device->createComputePipelineState(ps_desc);

	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list_ptr = cmd_list.Get();

	BufferDescription desc = {
		1024,
		BufferUsage::Storage
	};

	std::vector<float> buffer_init(256, 1.0f);
	BufferInitData buffer_init_data = {
		buffer_init.data(),
		buffer_init.size() * sizeof(float)
	};
	D3D12::Buffer x(device.get(), desc, &buffer_init_data, cmd_list_ptr);
	D3D12::Buffer y(device.get(), desc, &buffer_init_data, cmd_list_ptr);

	EXPECT_TRUE(x.handle());
	EXPECT_TRUE(y.handle());

	BufferDescription read_back_desc = {
		1024,
		BufferUsage::MapRead | BufferUsage::CopyDst
	};
	D3D12::Buffer read_back(device.get(), read_back_desc);
	EXPECT_TRUE(read_back.handle());

	// Descriptor
	Vcl::Graphics::D3D12::DescriptorTable table{ device.get(), &table_layout };
	table.addResource(0, &x, 0, 256, 4);
	table.addResource(1, &y, 0, 256, 4);

	// Dispatch computation
	cmd_list->SetPipelineState(pso.Get());
	table.setToCompute(cmd_list_ptr, 1);

	SaxpyKernelParameters kernel_params = { 16, 16, 2.0f };
	cmd_list->SetComputeRoot32BitConstants(0, 3, &kernel_params, 0);
	cmd_list->Dispatch(16, 16, 1);

	y.transition(cmd_list_ptr, D3D12_RESOURCE_STATE_COPY_SOURCE);

	cmd_list->CopyResource(read_back.handle(), y.handle());
	VCL_DIRECT3D_SAFE_CALL(cmd_list->Close());

	ID3D12CommandList* const generic_list = cmd_list_ptr;
	device->defaultQueue()->nativeQueue()->ExecuteCommandLists(1, &generic_list);
	device->defaultQueue()->sync();

	bool equal = true;
	auto ptr = (float*)read_back.map({ 0, 1024 });
	for (int i = 0; i < 256; i++)
	{
		equal = equal && (ptr[i] == 3.0f);
	}
	read_back.unmap();

	// Verify the result
	EXPECT_TRUE(equal);
}
