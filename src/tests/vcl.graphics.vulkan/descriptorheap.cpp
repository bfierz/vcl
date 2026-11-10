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
#include <utility>
#include <vector>

// Include the relevant parts from the library
#include <vcl/graphics/runtime/state/resourcetable.h>
#include <vcl/graphics/runtime/vulkan/resource/buffer.h>
#include <vcl/graphics/runtime/vulkan/resource/texture.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/descriptorheap.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/platform.h>

// Test
#include "test.h"

class VulkanDescriptorHeapTest : public VulkanTest
{
};

TEST(BindlessIndexAllocator, AllocatesDistinctIndicesUpToCapacity)
{
	using Vcl::Graphics::Runtime::BindlessIndexAllocator;

	BindlessIndexAllocator alloc{ 3 };
	EXPECT_EQ(alloc.capacity(), 3);
	EXPECT_EQ(alloc.allocate(), 0);
	EXPECT_EQ(alloc.allocate(), 1);
	EXPECT_EQ(alloc.allocate(), 2);
	EXPECT_EQ(alloc.nrAllocated(), 3);
	EXPECT_EQ(alloc.allocate(), BindlessIndexAllocator::InvalidIndex);
	EXPECT_EQ(alloc.nrAllocated(), 3);
}

TEST(BindlessIndexAllocator, RetiredSlotIsReusedAfterFenceCompleted)
{
	using Vcl::Graphics::Runtime::BindlessIndexAllocator;

	BindlessIndexAllocator alloc{ 2 };
	alloc.advance(1, 0);
	const uint32_t a = alloc.allocate();
	const uint32_t b = alloc.allocate();

	// Retired while recording the work of fence value 1
	alloc.retire(a);
	EXPECT_EQ(alloc.nrAllocated(), 1);
	EXPECT_EQ(alloc.nrRetired(), 1);
	EXPECT_EQ(alloc.allocate(), BindlessIndexAllocator::InvalidIndex) << "Slot is still in use by the GPU";

	// Fence value 1 is submitted, but not completed
	alloc.advance(2, 0);
	alloc.retire(b);
	EXPECT_EQ(alloc.allocate(), BindlessIndexAllocator::InvalidIndex);

	// Fence value 1 completed: only the first slot is available
	std::vector<uint32_t> freed;
	alloc.advance(3, 1, [&freed](uint32_t index) { freed.push_back(index); });
	EXPECT_EQ(freed, std::vector<uint32_t>{ a });
	EXPECT_EQ(alloc.allocate(), a);
	EXPECT_EQ(alloc.allocate(), BindlessIndexAllocator::InvalidIndex);

	alloc.advance(4, 2);
	EXPECT_EQ(alloc.nrRetired(), 0);
	EXPECT_EQ(alloc.allocate(), b);
}

TEST_F(VulkanDescriptorHeapTest, CreateHeap)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;

	BindlessLayoutDescription desc;
	desc.ResourceCapacity = 1024;
	desc.SamplerCapacity = 16;
	desc.PushConstantSize = 64;

	DescriptorHeap heap{ _context.get(), desc };
	EXPECT_NE(heap.layout(), VkPipelineLayout{ VK_NULL_HANDLE });
	EXPECT_EQ(heap.desc().ResourceCapacity, 1024);
	EXPECT_EQ(heap.nrResourceViews(), 0);
	EXPECT_EQ(heap.nrSamplers(), 0);
}

TEST_F(VulkanDescriptorHeapTest, CreateHeapWithStaticSamplers)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;

	SamplerDescription point;
	point.Filter = FilterType::MinMagMipPoint;
	SamplerDescription aniso;
	aniso.Filter = FilterType::Anisotropic;
	aniso.AddressU = TextureAddressMode::Wrap;
	aniso.AddressV = TextureAddressMode::Wrap;

	BindlessLayoutDescription desc;
	desc.StaticSamplers = { { 0, point }, { 3, aniso } };

	DescriptorHeap heap{ _context.get(), desc };
	EXPECT_NE(heap.layout(), VkPipelineLayout{ VK_NULL_HANDLE });
}

TEST_F(VulkanDescriptorHeapTest, BufferViews)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	const BufferDescription buffer_desc = { 1024, BufferUsage::Storage | BufferUsage::Uniform };
	Runtime::Vulkan::Buffer buffer(_context.get(), buffer_desc);

	DescriptorHeap heap{ _context.get(), {} };
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

TEST_F(VulkanDescriptorHeapTest, ViewsAreMoveOnlyHandles)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	const BufferDescription buffer_desc = { 1024, BufferUsage::Storage };
	Runtime::Vulkan::Buffer buffer(_context.get(), buffer_desc);

	DescriptorHeap heap{ _context.get(), {} };

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

TEST_F(VulkanDescriptorHeapTest, ReleasedSlotsAreReusedAfterTheFrameCompleted)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	const BufferDescription buffer_desc = { 1024, BufferUsage::Storage };
	Runtime::Vulkan::Buffer buffer(_context.get(), buffer_desc);

	BindlessLayoutDescription desc;
	desc.ResourceCapacity = 2;
	desc.SamplerCapacity = 1;
	DescriptorHeap heap{ _context.get(), desc };
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

TEST_F(VulkanDescriptorHeapTest, Samplers)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;

	DescriptorHeap heap{ _context.get(), {} };

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

TEST_F(VulkanDescriptorHeapTest, TextureViews)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	Texture2DDescription tex_desc;
	tex_desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	tex_desc.Usage = TextureUsage::Sampled | TextureUsage::Storage;
	tex_desc.Width = 16;
	tex_desc.Height = 16;
	tex_desc.MipLevels = 5;
	tex_desc.ArraySize = 4;
	Runtime::Vulkan::Texture2DArray tex{ _context.get(), tex_desc };

	Texture2DDescription depth_desc = tex_desc;
	depth_desc.Format = SurfaceFormat::D24_UNORM_S8_UINT;
	depth_desc.Usage = TextureUsage::Sampled | TextureUsage::OutputAttachment;
	depth_desc.MipLevels = 1;
	depth_desc.ArraySize = 1;
	Runtime::Vulkan::Texture2D depth{ _context.get(), depth_desc };

	DescriptorHeap heap{ _context.get(), {} };

	// Entire texture
	const ResourceView all = heap.createView(tex, ResourceViewType::Texture);
	EXPECT_TRUE(all);
	EXPECT_EQ(heap.viewType(all.index()), ResourceViewType::Texture);

	// Single layer of the lower mip-map levels, viewed as 2D texture
	TextureViewDescription layer;
	layer.Type = TextureType::Texture2D;
	layer.FirstMipLevel = 2;
	layer.FirstLayer = 3;
	layer.Layers = 1;
	const ResourceView single_layer = heap.createView(tex, ResourceViewType::Texture, layer);
	EXPECT_TRUE(single_layer);

	// Read-write views cover a single mip-map level
	TextureViewDescription level;
	level.FirstMipLevel = 1;
	level.MipLevels = 1;
	const ResourceView read_write = heap.createView(tex, ResourceViewType::RWTexture, level);
	EXPECT_TRUE(read_write);
	EXPECT_EQ(heap.viewType(read_write.index()), ResourceViewType::RWTexture);

	// Depth-stencil textures are sampled through their depth aspect
	const ResourceView depth_view = heap.createView(depth, ResourceViewType::Texture);
	EXPECT_TRUE(depth_view);

	EXPECT_EQ(heap.nrResourceViews(), 4);
}
