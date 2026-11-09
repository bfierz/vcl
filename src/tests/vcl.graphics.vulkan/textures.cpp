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
#include <cstring>
#include <memory>
#include <vector>

// Include the relevant parts from the library
#include <vcl/graphics/runtime/vulkan/resource/buffer.h>
#include <vcl/graphics/runtime/vulkan/resource/shader.h>
#include <vcl/graphics/runtime/vulkan/resource/texture.h>
#include <vcl/graphics/runtime/vulkan/state/pipelinestate.h>
#include <vcl/graphics/vulkan/commands.h>
#include <vcl/graphics/vulkan/descriptorset.h>

// Test
#include "test.h"

// Additional shaders
#include "colour.vert.spv.h"
#include "colour.frag.spv.h"

namespace {
	const stdext::span<const uint32_t> VertexShader32{ reinterpret_cast<uint32_t*>(ColourSpirvVSData), ColourSpirvVSSize / 4 };
	const stdext::span<const uint32_t> FragmentShader32{ reinterpret_cast<uint32_t*>(ColourSpirvFSData), ColourSpirvFSSize / 4 };

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

	std::vector<uint32_t> readBack(Vcl::Graphics::Runtime::Vulkan::Buffer& buffer)
	{
		std::vector<uint32_t> data(buffer.sizeInBytes() / sizeof(uint32_t));
		void* mapped = buffer.memory()->map(0, buffer.sizeInBytes());
		memcpy(data.data(), mapped, buffer.sizeInBytes());
		buffer.memory()->unmap();
		return data;
	}
}

class VulkanTextureTest : public VulkanTest
{
};

TEST_F(VulkanTextureTest, UploadAndCopyToBuffer)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	constexpr uint32_t width = 16;
	constexpr uint32_t height = 8;

	CommandQueue queue{ _context.get(), VK_QUEUE_GRAPHICS_BIT, 0 };
	CommandBuffer cmd_buffer{ *_context, _context->commandPool(0, CommandBufferType::Transient) };

	std::vector<uint32_t> pixels(width * height);
	for (uint32_t i = 0; i < pixels.size(); i++)
		pixels[i] = 0xff000000 | i;

	TextureResource init_data;
	init_data.Data = { reinterpret_cast<const unsigned char*>(pixels.data()), pixels.size() * sizeof(uint32_t) };
	init_data.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_data.Width = width;
	init_data.Height = height;

	BufferDescription read_desc = { width * height * 4, BufferUsage::MapRead | BufferUsage::CopyDst };
	Runtime::Vulkan::Buffer read_buffer{ _context.get(), read_desc };

	cmd_buffer.begin();
	Runtime::Vulkan::Texture2D tex{ _context.get(), makeTexture2DDescription(SurfaceFormat::R8G8B8A8_UNORM, TextureUsage::Sampled | TextureUsage::CopySrc, width, height), &init_data, cmd_buffer };
	tex.copyTo(cmd_buffer, read_buffer);
	cmd_buffer.end();

	std::array<VkCommandBuffer, 1> cmd_buffers = { cmd_buffer };
	queue.submit(stdext::make_span(cmd_buffers), VK_NULL_HANDLE);
	queue.waitIdle();

	EXPECT_EQ(tex.type(), TextureType::Texture2D);
	EXPECT_EQ(tex.layout(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	EXPECT_EQ(readBack(read_buffer), pixels);
}

TEST_F(VulkanTextureTest, CreateTextureTypes)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture2DDescription desc_2d = makeTexture2DDescription(SurfaceFormat::R8G8B8A8_UNORM, TextureUsage::Sampled, 8, 8);
	desc_2d.ArraySize = 4;
	Runtime::Vulkan::Texture2DArray tex_2d_array{ _context.get(), desc_2d };
	EXPECT_EQ(tex_2d_array.type(), TextureType::Texture2DArray);
	EXPECT_EQ(tex_2d_array.layers(), 4);

	TextureCubeDescription desc_cube;
	desc_cube.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc_cube.Usage = TextureUsage::Sampled;
	desc_cube.Width = 8;
	desc_cube.Height = 8;
	desc_cube.MipLevels = 2;
	desc_cube.ArraySize = 1;
	Runtime::Vulkan::TextureCube tex_cube{ _context.get(), desc_cube };
	EXPECT_EQ(tex_cube.type(), TextureType::TextureCube);
	EXPECT_EQ(tex_cube.layers(), 6);
	EXPECT_EQ(tex_cube.mipMapLevels(), 2);

	Texture3DDescription desc_3d;
	desc_3d.Format = SurfaceFormat::R32_FLOAT;
	desc_3d.Usage = TextureUsage::Storage;
	desc_3d.Width = 4;
	desc_3d.Height = 4;
	desc_3d.Depth = 4;
	desc_3d.MipLevels = 1;
	Runtime::Vulkan::Texture3D tex_3d{ _context.get(), desc_3d };
	EXPECT_EQ(tex_3d.type(), TextureType::Texture3D);
	EXPECT_EQ(tex_3d.depth(), 4);
	EXPECT_NE(tex_3d.view(), VkImageView{ VK_NULL_HANDLE });

	Runtime::Vulkan::Texture2D depth{ _context.get(), makeTexture2DDescription(SurfaceFormat::D32_FLOAT, TextureUsage::OutputAttachment, 8, 8) };
	EXPECT_EQ(depth.aspects(), VkImageAspectFlags{ VK_IMAGE_ASPECT_DEPTH_BIT });
}

TEST_F(VulkanTextureTest, RenderTriangle)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	constexpr uint32_t width = 32;
	constexpr uint32_t height = 32;

	CommandQueue queue{ _context.get(), VK_QUEUE_GRAPHICS_BIT, 0 };
	CommandBuffer cmd_buffer{ *_context, _context->commandPool(0, CommandBufferType::Transient) };

	// Pipeline drawing coloured, transformed triangles
	Runtime::Vulkan::Shader vs(*_context, ShaderType::VertexShader, 0, VertexShader32);
	Runtime::Vulkan::Shader fs(*_context, ShaderType::FragmentShader, 0, FragmentShader32);
	DescriptorSetLayout descriptor_set_layout{ _context.get(), {} };
	Runtime::Vulkan::PipelineLayout layout{ _context.get(), &descriptor_set_layout, {} };

	PipelineStateDescription desc;
	desc.VertexShader = &vs;
	desc.FragmentShader = &fs;
	desc.InputAssembly.Topology = PrimitiveType::Trianglelist;
	desc.InputLayout = {
		{
			{ 0, sizeof(float) * 5, VertexDataClassification::VertexDataPerObject },
			{ 1, sizeof(float) * 16, VertexDataClassification::VertexDataPerInstance },
		},
		{
			{ "Position", SurfaceFormat::R32G32_FLOAT, 0, 0, 0 },
			{ "Colour", SurfaceFormat::R32G32B32_FLOAT, 0, 0, sizeof(float) * 2 },
			{ "Transform", SurfaceFormat::R32G32B32A32_FLOAT, 4, 1, 0 },
		}
	};
	desc.Rasterizer.CullMode = CullModeMethod::None;
	desc.DepthStencil.DepthEnable = false;

	RenderTargetLayout rt_layout;
	rt_layout.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
	rt_layout.DepthStencilFormat = SurfaceFormat::Unknown;
	Runtime::Vulkan::GraphicsPipelineState pps{ _context.get(), &layout, desc, rt_layout };

	// Red triangle covering the upper-left half of the render-target
	const std::array<float, 15> vertices = {
		-1,
		-1,
		1,
		0,
		0,
		1,
		-1,
		1,
		0,
		0,
		-1,
		1,
		1,
		0,
		0,
	};
	const BufferInitData vertex_data = { vertices.data(), sizeof(vertices) };
	Runtime::Vulkan::Buffer vertex_buffer{ _context.get(), { sizeof(vertices), BufferUsage::Vertex | BufferUsage::MapWrite }, &vertex_data };

	const std::array<float, 16> transform = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
	const BufferInitData instance_data = { transform.data(), sizeof(transform) };
	Runtime::Vulkan::Buffer instance_buffer{ _context.get(), { sizeof(transform), BufferUsage::Vertex | BufferUsage::MapWrite }, &instance_data };

	BufferDescription read_desc = { width * height * 4, BufferUsage::MapRead | BufferUsage::CopyDst };
	Runtime::Vulkan::Buffer read_buffer{ _context.get(), read_desc };

	auto rt = std::make_unique<Runtime::Vulkan::Texture2D>(_context.get(), makeTexture2DDescription(SurfaceFormat::R8G8B8A8_UNORM, TextureUsage::OutputAttachment | TextureUsage::CopySrc, width, height));

	RenderPassDescription pass;
	RenderTargetAttachmentDescription rt_attachment;
	rt_attachment.View = rt;
	rt_attachment.LoadOp = AttachmentLoadOp::Clear;
	rt_attachment.StoreOp = AttachmentStoreOp::Store;
	rt_attachment.ClearColor = { 0, 0, 1, 1 };
	pass.RenderTargetAttachments.push_back(rt_attachment);

	cmd_buffer.begin();
	cmd_buffer.beginRendering(pass);
	VkViewport viewport = { 0, 0, width, height, 0, 1 };
	VkRect2D scissor = { { 0, 0 }, { width, height } };
	cmd_buffer.setViewport(0, { &viewport, 1 });
	cmd_buffer.setScissor(0, { &scissor, 1 });
	cmd_buffer.bind(VK_PIPELINE_BIND_POINT_GRAPHICS, pps);
	cmd_buffer.bindVertexBuffer(0, vertex_buffer.id());
	cmd_buffer.bindVertexBuffer(1, instance_buffer.id());
	cmd_buffer.draw(3);
	cmd_buffer.endRendering();
	rt->copyTo(cmd_buffer, read_buffer);
	cmd_buffer.end();

	std::array<VkCommandBuffer, 1> cmd_buffers = { cmd_buffer };
	queue.submit(stdext::make_span(cmd_buffers), VK_NULL_HANDLE);
	queue.waitIdle();

	// R8G8B8A8_UNORM as little-endian 32-bit values
	constexpr uint32_t red = 0xff0000ff;
	constexpr uint32_t blue = 0xffff0000;
	const auto pixels = readBack(read_buffer);
	EXPECT_EQ(pixels[2 * width + 2], red) << "Pixel inside the triangle";
	EXPECT_EQ(pixels[(height - 3) * width + (width - 3)], blue) << "Pixel outside the triangle";
}
