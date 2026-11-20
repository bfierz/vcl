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
#include <vcl/graphics/d3d12/descriptorheap.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/d3d12/resource/shader.h>
#include <vcl/graphics/runtime/d3d12/state/pipelinestate.h>

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

	// Heap index of 'x'
	uint32_t X;

	// Heap index of 'y'
	uint32_t Y;
};

TEST(D3D12Compute, Saxpy)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;

	if (!device->features().BindlessResources)
		GTEST_SKIP() << "Bindless resources need shader model 6.6 and resource binding tier 3.";

	DescriptorHeap heap{ device.get(), {} };

	D3D12::Shader cs(ShaderType::ComputeShader, 0, SaxpyCsoCS);
	ComputePipelineStateDescription cpd;
	cpd.ComputeShader = &cs;
	D3D12::ComputePipelineState cps{ device.get(), &heap, cpd };

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

	// 'x' is a structured buffer, 'y' a raw buffer
	const ResourceView x_view = heap.createView(x, ResourceViewType::Buffer, 0, 1024, sizeof(float));
	const ResourceView y_view = heap.createView(y, ResourceViewType::RWBuffer, 0, 1024);

	x.transition(cmd_list_ptr, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	y.transition(cmd_list_ptr, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	// Dispatch computation
	heap.bind(cmd_list_ptr);
	cmd_list->SetPipelineState(cps.handle());

	const SaxpyKernelParameters kernel_params = { 16, 16, 2.0f, x_view.index().Value, y_view.index().Value };
	cmd_list->SetComputeRoot32BitConstants(DescriptorHeap::RootConstantsParameter, sizeof(kernel_params) / 4, &kernel_params, 0);
	cmd_list->Dispatch(1, 1, 1);

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
