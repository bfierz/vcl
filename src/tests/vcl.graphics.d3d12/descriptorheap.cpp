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

// VCL configuration
#include <vcl/config/global.h>

// C++ Standard Library
#include <array>
#include <chrono>
#include <cstring>
#include <utility>
#include <vector>

// Include the relevant parts from the library
#include <vcl/graphics/d3d12/d3d.h>
#include <directx/d3dx12.h>
#include <vcl/graphics/d3d12/descriptorheap.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/d3d12/resource/shader.h>
#include <vcl/graphics/runtime/d3d12/resource/texture.h>
#include <vcl/graphics/runtime/d3d12/state/pipelinestate.h>

// Google test
#include <gtest/gtest.h>

#include "bindless.ps.hlsl.cso.h"
#include "fullscreen.vs.hlsl.cso.h"
#include "gradient.cs.hlsl.cso.h"

extern std::unique_ptr<Vcl::Graphics::D3D12::Device> device;

namespace {
	Vcl::Graphics::Runtime::Texture2DDescription makeTexture2DDescription(Vcl::Graphics::SurfaceFormat format, Vcl::Flags<Vcl::Graphics::Runtime::TextureUsage> usage, uint32_t width, uint32_t height)
	{
		Vcl::Graphics::Runtime::Texture2DDescription desc;
		desc.Format = format;
		desc.Usage = usage;
		desc.Width = width;
		desc.Height = height;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		return desc;
	}

	//! Read back a buffer of 32-bit values
	std::vector<uint32_t> readBack(const Vcl::Graphics::Runtime::D3D12::Buffer& buffer)
	{
		std::vector<uint32_t> data(buffer.sizeInBytes() / sizeof(uint32_t));
		const void* mapped = buffer.map({ 0, buffer.sizeInBytes() });
		std::memcpy(data.data(), mapped, data.size() * sizeof(uint32_t));
		buffer.unmap();
		return data;
	}

	//! Submit a command list and wait until the GPU executed it
	void submitAndWait(ID3D12GraphicsCommandList* cmd_list)
	{
		VCL_DIRECT3D_SAFE_CALL(cmd_list->Close());
		ID3D12CommandList* const lists[] = { cmd_list };
		device->defaultQueue()->nativeQueue()->ExecuteCommandLists(1, lists);
		ASSERT_TRUE(device->defaultQueue()->sync(std::chrono::seconds{ 10 }));
	}

	//! Parameters of 'bindless.ps.hlsl'
	struct BindlessParameters
	{
		uint32_t Constants;
		uint32_t Texture;
		uint32_t Sampler;
		uint32_t UseStaticSampler;
	};

	//! Parameters of 'gradient.cs.hlsl'
	struct GradientParameters
	{
		uint32_t Output;
		uint32_t Width;
		uint32_t Height;
	};
}

class D3D12DescriptorHeapTest : public testing::Test
{
public:
	void SetUp() override
	{
		if (!device->features().BindlessResources)
			GTEST_SKIP() << "Bindless resources need shader model 6.6 and resource binding tier 3.";
	}
};

TEST(D3D12DeviceFeatures, BindlessNeedsShaderModel66AndBindingTier3)
{
	const bool supported = device->shaderModel() >= D3D_SHADER_MODEL_6_6 && device->resourceBindingTier() >= D3D12_RESOURCE_BINDING_TIER_3;
	EXPECT_EQ(device->features().BindlessResources, supported);

	const auto& features = device->features();
	EXPECT_EQ(features.MaxPushConstantSize, 256);
	EXPECT_EQ(features.MaxResourceHeapCapacity, 1000000);
	EXPECT_EQ(features.MaxSamplerHeapCapacity, 2048);
	EXPECT_EQ(features.MaxRenderTargets, 8);

	// The default layout fits into the limits
	EXPECT_EQ(Vcl::Graphics::Runtime::validate(Vcl::Graphics::Runtime::BindlessLayoutDescription{}, features).empty(), supported);
}

TEST_F(D3D12DescriptorHeapTest, CreateHeap)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;

	BindlessLayoutDescription desc;
	desc.ResourceCapacity = 1024;
	desc.SamplerCapacity = 16;
	desc.PushConstantSize = 64;

	DescriptorHeap heap{ device.get(), desc };
	EXPECT_NE(heap.rootSignature(), nullptr);
	EXPECT_NE(heap.resourceHeap(), nullptr);
	EXPECT_NE(heap.samplerHeap(), nullptr);
	EXPECT_EQ(heap.desc().ResourceCapacity, 1024);
	EXPECT_EQ(heap.nrResourceViews(), 0);
	EXPECT_EQ(heap.nrSamplers(), 0);

	EXPECT_EQ(heapDesc(heap.resourceHeap()).NumDescriptors, 1024);
	EXPECT_EQ(heapDesc(heap.samplerHeap()).NumDescriptors, 16);
	EXPECT_EQ(heapDesc(heap.resourceHeap()).Flags, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
}

TEST_F(D3D12DescriptorHeapTest, CreateHeapWithStaticSamplers)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;

	SamplerDescription point;
	point.Filter = FilterType::MinMagMipPoint;
	SamplerDescription aniso;
	aniso.Filter = FilterType::Anisotropic;
	aniso.AddressU = TextureAddressMode::Wrap;
	aniso.AddressV = TextureAddressMode::Wrap;

	BindlessLayoutDescription desc;
	desc.StaticSamplers = { { 0, point }, { 3, aniso } };

	DescriptorHeap heap{ device.get(), desc };
	EXPECT_NE(heap.rootSignature(), nullptr);
}

TEST_F(D3D12DescriptorHeapTest, BufferViews)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;
	using namespace Vcl::Graphics;

	const BufferDescription buffer_desc = { 1024, BufferUsage::Storage | BufferUsage::Uniform };
	Runtime::D3D12::Buffer buffer(device.get(), buffer_desc);

	DescriptorHeap heap{ device.get(), {} };
	{
		const ResourceView constants = heap.createView(buffer, ResourceViewType::ConstantBuffer, 0, 256);
		const ResourceView raw = heap.createView(buffer, ResourceViewType::Buffer, 256, 256);
		const ResourceView structured = heap.createView(buffer, ResourceViewType::RWBuffer, 512, 512, 16);

		EXPECT_TRUE(constants);
		EXPECT_TRUE(raw);
		EXPECT_TRUE(structured);
		EXPECT_NE(constants.index().Value, raw.index().Value);
		EXPECT_NE(constants.index().Value, structured.index().Value);
		EXPECT_NE(raw.index().Value, structured.index().Value);

		EXPECT_EQ(heap.viewType(constants.index()), ResourceViewType::ConstantBuffer);
		EXPECT_EQ(heap.viewType(raw.index()), ResourceViewType::Buffer);
		EXPECT_EQ(heap.viewType(structured.index()), ResourceViewType::RWBuffer);
		EXPECT_EQ(heap.nrResourceViews(), 3);
	}
	EXPECT_EQ(heap.nrResourceViews(), 0);
}

TEST_F(D3D12DescriptorHeapTest, SmallConstantBufferIsPaddedTo256Bytes)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;
	using namespace Vcl::Graphics;

	// Constant buffer views cover multiples of 256 bytes
	const BufferDescription buffer_desc = { 16, BufferUsage::Uniform | BufferUsage::MapWrite };
	Runtime::D3D12::Buffer buffer(device.get(), buffer_desc);
	EXPECT_EQ(buffer.sizeInBytes(), 16);
	EXPECT_EQ(resourceDesc(buffer.handle()).Width, 256);

	DescriptorHeap heap{ device.get(), {} };
	const ResourceView view = heap.createView(buffer, ResourceViewType::ConstantBuffer, 0, 16);
	EXPECT_TRUE(view);
}

TEST_F(D3D12DescriptorHeapTest, ViewsAreMoveOnlyHandles)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;
	using namespace Vcl::Graphics;

	const BufferDescription buffer_desc = { 1024, BufferUsage::Storage };
	Runtime::D3D12::Buffer buffer(device.get(), buffer_desc);

	DescriptorHeap heap{ device.get(), {} };

	ResourceView empty;
	EXPECT_FALSE(empty);
	EXPECT_FALSE(empty.index().isValid());

	ResourceView a = heap.createView(buffer, ResourceViewType::Buffer, 0, 1024);
	const uint32_t index = a.index().Value;

	ResourceView b = std::move(a);
	EXPECT_FALSE(a);
	EXPECT_EQ(b.index().Value, index);
	EXPECT_EQ(heap.nrResourceViews(), 1);

	// Assigning releases the view held before
	ResourceView c = heap.createView(buffer, ResourceViewType::Buffer, 0, 512);
	EXPECT_EQ(heap.nrResourceViews(), 2);
	c = std::move(b);
	EXPECT_EQ(c.index().Value, index);
	EXPECT_EQ(heap.nrResourceViews(), 1);

	c.reset();
	EXPECT_FALSE(c);
	EXPECT_EQ(heap.nrResourceViews(), 0);
}

TEST_F(D3D12DescriptorHeapTest, ReleasedSlotsAreReusedAfterTheFrameCompleted)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;
	using namespace Vcl::Graphics;

	const BufferDescription buffer_desc = { 1024, BufferUsage::Storage };
	Runtime::D3D12::Buffer buffer(device.get(), buffer_desc);

	BindlessLayoutDescription desc;
	desc.ResourceCapacity = 2;
	desc.SamplerCapacity = 1;
	DescriptorHeap heap{ device.get(), desc };
	heap.advance(1, 0);

	ResourceView a = heap.createView(buffer, ResourceViewType::Buffer, 0, 1024);
	ResourceView b = heap.createView(buffer, ResourceViewType::Buffer, 0, 1024);
	SamplerView s = heap.createSampler({});
	const uint32_t index_a = a.index().Value;
	const uint32_t index_s = s.index().Value;

	// Released while recording frame 1; the slots can't be reused before it completed
	a.reset();
	s.reset();
	heap.advance(2, 0);
	EXPECT_EQ(heap.nrResourceViews(), 1);
	EXPECT_EQ(heap.nrSamplers(), 0);

	heap.advance(3, 1);
	a = heap.createView(buffer, ResourceViewType::RWBuffer, 0, 1024);
	s = heap.createSampler({});
	EXPECT_EQ(a.index().Value, index_a);
	EXPECT_EQ(heap.viewType(a.index()), ResourceViewType::RWBuffer);
	EXPECT_EQ(s.index().Value, index_s);
}

TEST_F(D3D12DescriptorHeapTest, Samplers)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;

	DescriptorHeap heap{ device.get(), {} };

	SamplerDescription shadow;
	shadow.Filter = FilterType::ComparisonMinMagLinearMipPoint;
	shadow.ComparisonFunc = ComparisonFunction::LessEqual;
	shadow.AddressU = TextureAddressMode::Border;
	shadow.AddressV = TextureAddressMode::Border;
	shadow.BorderColor = { 1, 1, 1, 1 };

	const SamplerView linear = heap.createSampler({});
	const SamplerView compare = heap.createSampler(shadow);
	EXPECT_TRUE(linear);
	EXPECT_TRUE(compare);
	EXPECT_NE(linear.index().Value, compare.index().Value);
	EXPECT_EQ(heap.nrSamplers(), 2);
}

TEST_F(D3D12DescriptorHeapTest, TextureViews)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;
	using namespace Vcl::Graphics;

	Texture2DDescription tex_desc;
	tex_desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	tex_desc.Usage = TextureUsage::Sampled | TextureUsage::Storage;
	tex_desc.Width = 16;
	tex_desc.Height = 16;
	tex_desc.MipLevels = 5;
	tex_desc.ArraySize = 4;
	Runtime::D3D12::Texture2DArray tex{ device.get(), tex_desc };
	EXPECT_EQ(tex.type(), TextureType::Texture2DArray);

	Texture2DDescription depth_desc = tex_desc;
	depth_desc.Format = SurfaceFormat::D24_UNORM_S8_UINT;
	depth_desc.Usage = TextureUsage::Sampled | TextureUsage::OutputAttachment;
	depth_desc.MipLevels = 1;
	depth_desc.ArraySize = 1;
	Runtime::D3D12::Texture2D depth{ device.get(), depth_desc };

	// Sampled depth textures are typeless, so that shaders can read the depth part
	EXPECT_EQ(resourceDesc(depth.handle()).Format, DXGI_FORMAT_R24G8_TYPELESS);
	EXPECT_TRUE(resourceDesc(depth.handle()).Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);

	DescriptorHeap heap{ device.get(), {} };

	// Entire texture
	const ResourceView all = heap.createView(tex, ResourceViewType::Texture);
	EXPECT_TRUE(all);
	EXPECT_EQ(heap.viewType(all.index()), ResourceViewType::Texture);

	// Single layer of the lower mip-map levels. D3D12 views other than the
	// first layer as array.
	TextureViewDescription layer;
	layer.FirstMipLevel = 2;
	layer.FirstLayer = 3;
	layer.Layers = 1;
	const ResourceView single_layer = heap.createView(tex, ResourceViewType::Texture, layer);
	EXPECT_TRUE(single_layer);

	// The first layer viewed as 2D texture
	TextureViewDescription first_layer;
	first_layer.Type = TextureType::Texture2D;
	first_layer.Layers = 1;
	const ResourceView first_layer_2d = heap.createView(tex, ResourceViewType::Texture, first_layer);
	EXPECT_TRUE(first_layer_2d);

	// Read-write views cover a single mip-map level
	TextureViewDescription level;
	level.FirstMipLevel = 1;
	level.MipLevels = 1;
	const ResourceView read_write = heap.createView(tex, ResourceViewType::RWTexture, level);
	EXPECT_TRUE(read_write);
	EXPECT_EQ(heap.viewType(read_write.index()), ResourceViewType::RWTexture);

	// Depth-stencil textures are sampled through their depth part
	const ResourceView depth_view = heap.createView(depth, ResourceViewType::Texture);
	EXPECT_TRUE(depth_view);

	EXPECT_EQ(heap.nrResourceViews(), 5);
}

TEST_F(D3D12DescriptorHeapTest, TextureViewReinterpretingTheFormat)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;
	using namespace Vcl::Graphics;

	// Reinterpreting the format is declared when the texture is created
	Texture2DDescription tex_desc;
	tex_desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	tex_desc.ViewFormats = { SurfaceFormat::R8G8B8A8_UNORM_SRGB };
	tex_desc.Usage = TextureUsage::Sampled;
	tex_desc.Width = 16;
	tex_desc.Height = 16;
	tex_desc.MipLevels = 1;
	tex_desc.ArraySize = 1;
	Runtime::D3D12::Texture2D tex{ device.get(), tex_desc };

	EXPECT_TRUE(tex.supportsViewFormat(SurfaceFormat::Unknown));
	EXPECT_TRUE(tex.supportsViewFormat(SurfaceFormat::R8G8B8A8_UNORM));
	EXPECT_TRUE(tex.supportsViewFormat(SurfaceFormat::R8G8B8A8_UNORM_SRGB));
	EXPECT_FALSE(tex.supportsViewFormat(SurfaceFormat::R32_FLOAT));

	// The resource is created typeless
	EXPECT_EQ(resourceDesc(tex.handle()).Format, DXGI_FORMAT_R8G8B8A8_TYPELESS);

	DescriptorHeap heap{ device.get(), {} };

	TextureViewDescription srgb;
	srgb.Format = SurfaceFormat::R8G8B8A8_UNORM_SRGB;
	const ResourceView linear_view = heap.createView(tex, ResourceViewType::Texture);
	const ResourceView srgb_view = heap.createView(tex, ResourceViewType::Texture, srgb);
	EXPECT_TRUE(linear_view);
	EXPECT_TRUE(srgb_view);
}

// Sample a texture through the heaps, once with a sampler from the sampler
// heap and once with a static sampler
TEST_F(D3D12DescriptorHeapTest, RenderBindlessTexture)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;
	using namespace Vcl::Graphics;

	// Rows of 256 bytes need no padding in the read-back buffer
	constexpr uint32_t width = 64;
	constexpr uint32_t height = 64;

	// R8G8B8A8_UNORM as little-endian 32-bit values
	constexpr uint32_t black = 0xff000000;
	constexpr uint32_t red = 0xff0000ff;
	constexpr uint32_t green = 0xff00ff00;
	constexpr uint32_t blue = 0xffff0000;
	constexpr uint32_t yellow = 0xff00ffff;
	constexpr uint32_t white = 0xffffffff;

	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);

	// Layout with a static point sampler in register 0
	SamplerDescription point_sampler;
	point_sampler.Filter = FilterType::MinMagMipPoint;
	BindlessLayoutDescription heap_desc;
	heap_desc.StaticSamplers = { { 0, point_sampler } };
	DescriptorHeap heap{ device.get(), heap_desc };

	Runtime::D3D12::Shader vs(ShaderType::VertexShader, 0, FullscreenCsoVS);
	Runtime::D3D12::Shader ps(ShaderType::FragmentShader, 0, BindlessCsoPS);

	PipelineStateDescription desc;
	desc.VertexShader = &vs;
	desc.FragmentShader = &ps;
	desc.InputAssembly.Topology = PrimitiveType::Trianglelist;
	desc.Rasterizer.CullMode = CullModeMethod::None;
	desc.DepthStencil.DepthEnable = false;

	RenderTargetLayout rt_layout;
	rt_layout.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
	rt_layout.DepthStencilFormat = SurfaceFormat::Unknown;
	desc.RenderTargets = rt_layout;
	Runtime::D3D12::GraphicsPipelineState pps{ device.get(), &heap, desc };
	EXPECT_EQ(pps.rootSignature(), heap.rootSignature());

	// Constant buffer removing the blue channel
	const std::array<float, 4> tint = { 1, 1, 0, 1 };
	Runtime::D3D12::Buffer constants{ device.get(), { sizeof(tint), BufferUsage::Uniform | BufferUsage::MapWrite } };
	std::memcpy(constants.map(), tint.data(), sizeof(tint));
	constants.unmap();

	BufferDescription read_desc = { width * height * 4, BufferUsage::MapRead | BufferUsage::CopyDst };
	Runtime::D3D12::Buffer read_buffer_heap{ device.get(), read_desc };
	Runtime::D3D12::Buffer read_buffer_static{ device.get(), read_desc };

	// 2x2 texture with a different colour in each quadrant
	const std::array<uint32_t, 4> texels = { red, green, blue, white };
	TextureResource init_data;
	init_data.Data = { reinterpret_cast<const unsigned char*>(texels.data()), sizeof(texels) };
	init_data.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_data.Width = 2;
	init_data.Height = 2;
	Runtime::D3D12::Texture2D tex{ device.get(), makeTexture2DDescription(SurfaceFormat::R8G8B8A8_UNORM, TextureUsage::Sampled, 2, 2), &init_data, cmd_list.Get() };
	tex.transition(cmd_list.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

	// Clearing to a different value than declared triggers a debug-layer warning
	const TextureClearValue clear_value = { { 1, 0, 1, 1 } };
	auto rt_desc = makeTexture2DDescription(SurfaceFormat::R8G8B8A8_UNORM, TextureUsage::OutputAttachment | TextureUsage::CopySrc, width, height);
	rt_desc.ClearValue = clear_value;
	Runtime::D3D12::Texture2D rt{ device.get(), rt_desc };
	auto rtv_heap = device->createDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, false);
	const auto rtv = cpuHeapStart(rtv_heap.Get());
	device->nativeDevice()->CreateRenderTargetView(rt.handle(), nullptr, rtv);

	// Occupy the first slots, such that the shader has to use the indices
	const ResourceView unused_view = heap.createView(constants, ResourceViewType::ConstantBuffer, 0, sizeof(tint));
	const SamplerView unused_sampler = heap.createSampler({});

	const ResourceView constants_view = heap.createView(constants, ResourceViewType::ConstantBuffer, 0, sizeof(tint));
	const ResourceView tex_view = heap.createView(tex, ResourceViewType::Texture);
	const SamplerView sampler = heap.createSampler(point_sampler);

	const D3D12_VIEWPORT viewport = { 0, 0, width, height, 0, 1 };
	const D3D12_RECT scissor = { 0, 0, width, height };
	heap.bind(cmd_list.Get());
	for (const uint32_t use_static_sampler : { 0u, 1u })
	{
		rt.transition(cmd_list.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET);
		cmd_list->ClearRenderTargetView(rtv, clear_value.Colour.data(), 0, nullptr);
		cmd_list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
		cmd_list->RSSetViewports(1, &viewport);
		cmd_list->RSSetScissorRects(1, &scissor);
		cmd_list->SetPipelineState(pps.handle());
		cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		const BindlessParameters params = { constants_view.index().Value, tex_view.index().Value, sampler.index().Value, use_static_sampler };
		cmd_list->SetGraphicsRoot32BitConstants(DescriptorHeap::RootConstantsParameter, sizeof(params) / 4, &params, 0);
		cmd_list->DrawInstanced(3, 1, 0, 0);

		rt.copyTo(cmd_list.Get(), use_static_sampler ? read_buffer_static : read_buffer_heap);
	}
	submitAndWait(cmd_list.Get());

	for (auto* read_buffer : { &read_buffer_heap, &read_buffer_static })
	{
		const auto pixels = readBack(*read_buffer);
		EXPECT_EQ(pixels[16 * width + 16], red) << "Upper-left quadrant";
		EXPECT_EQ(pixels[16 * width + 48], green) << "Upper-right quadrant";
		EXPECT_EQ(pixels[48 * width + 16], black) << "Lower-left quadrant: blue without the blue channel";
		EXPECT_EQ(pixels[48 * width + 48], yellow) << "Lower-right quadrant: white without the blue channel";
	}
}

// Write a texture through a read-write view of the heap
TEST_F(D3D12DescriptorHeapTest, WriteReadWriteTexture)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::D3D12;
	using namespace Vcl::Graphics;

	constexpr uint32_t width = 64;
	constexpr uint32_t height = 64;

	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);

	DescriptorHeap heap{ device.get(), {} };

	Runtime::D3D12::Shader cs(ShaderType::ComputeShader, 0, GradientCsoCS);
	ComputePipelineStateDescription cpd;
	cpd.ComputeShader = &cs;
	Runtime::D3D12::ComputePipelineState cps{ device.get(), &heap, cpd };

	Runtime::D3D12::Texture2D tex{ device.get(), makeTexture2DDescription(SurfaceFormat::R8G8B8A8_UNORM, TextureUsage::Storage | TextureUsage::CopySrc, width, height) };
	Runtime::D3D12::Buffer read_buffer{ device.get(), { width * height * 4, BufferUsage::MapRead | BufferUsage::CopyDst } };

	// Occupy the first slot, such that the shader has to use the index
	Runtime::D3D12::Buffer unused{ device.get(), { 256, BufferUsage::Storage } };
	const ResourceView unused_view = heap.createView(unused, ResourceViewType::Buffer, 0, 256);
	const ResourceView output = heap.createView(tex, ResourceViewType::RWTexture);
	EXPECT_NE(output.index().Value, 0);

	tex.transition(cmd_list.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	heap.bind(cmd_list.Get());
	cmd_list->SetPipelineState(cps.handle());
	const GradientParameters params = { output.index().Value, width, height };
	cmd_list->SetComputeRoot32BitConstants(DescriptorHeap::RootConstantsParameter, sizeof(params) / 4, &params, 0);
	cmd_list->Dispatch(width / 8, height / 8, 1);
	tex.copyTo(cmd_list.Get(), read_buffer);
	submitAndWait(cmd_list.Get());

	const auto pixels = readBack(read_buffer);
	bool equal = true;
	for (uint32_t y = 0; y < height; y++)
		for (uint32_t x = 0; x < width; x++)
			equal = equal && pixels[y * width + x] == (0xff000000u | (y << 8) | x);
	EXPECT_TRUE(equal);
	EXPECT_EQ(pixels[5 * width + 7], 0xff000507u);
}
