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

// Include the relevant parts from the library
#include <vcl/graphics/runtime/vulkan/resource/shader.h>
#include <vcl/graphics/runtime/vulkan/state/pipelinestate.h>
#include <vcl/graphics/vulkan/descriptorset.h>

// Test
#include "test.h"

// Additional shaders
#include "colour.vert.spv.h"
#include "colour.frag.spv.h"
const stdext::span<const uint32_t> ColourSpirvVS32{ reinterpret_cast<uint32_t*>(ColourSpirvVSData), ColourSpirvVSSize / 4 };
const stdext::span<const uint32_t> ColourSpirvFS32{ reinterpret_cast<uint32_t*>(ColourSpirvFSData), ColourSpirvFSSize / 4 };

class VulkanPipelineStateTest : public VulkanTest
{
protected:
	Vcl::Graphics::Runtime::PipelineStateDescription createDescription(Vcl::Graphics::Runtime::Shader* vs, Vcl::Graphics::Runtime::Shader* fs)
	{
		using namespace Vcl::Graphics::Runtime;
		using Vcl::Graphics::SurfaceFormat;

		PipelineStateDescription desc;
		desc.VertexShader = vs;
		desc.FragmentShader = fs;
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

		return desc;
	}
};

TEST_F(VulkanPipelineStateTest, CreateGraphicsPipeline)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	Runtime::Vulkan::Shader vs(*_context, ShaderType::VertexShader, 0, ColourSpirvVS32);
	Runtime::Vulkan::Shader fs(*_context, ShaderType::FragmentShader, 0, ColourSpirvFS32);

	DescriptorSetLayout descriptor_set_layout{ _context.get(), {} };
	Runtime::Vulkan::PipelineLayout layout{ _context.get(), &descriptor_set_layout, {} };

	RenderTargetLayout rt_layout;
	rt_layout.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
	rt_layout.DepthStencilFormat = SurfaceFormat::D32_FLOAT;

	auto desc = createDescription(&vs, &fs);
	Runtime::Vulkan::GraphicsPipelineState pps{ _context.get(), &layout, desc, rt_layout };
	EXPECT_NE(static_cast<VkPipeline>(pps), VkPipeline{ VK_NULL_HANDLE });
}

TEST_F(VulkanPipelineStateTest, CreateGraphicsPipelineWithFixedFunctionState)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	Runtime::Vulkan::Shader vs(*_context, ShaderType::VertexShader, 0, ColourSpirvVS32);
	Runtime::Vulkan::Shader fs(*_context, ShaderType::FragmentShader, 0, ColourSpirvFS32);

	DescriptorSetLayout descriptor_set_layout{ _context.get(), {} };
	Runtime::Vulkan::PipelineLayout layout{ _context.get(), &descriptor_set_layout, {} };

	RenderTargetLayout rt_layout;
	rt_layout.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM, SurfaceFormat::R16G16B16A16_FLOAT };
	rt_layout.DepthStencilFormat = SurfaceFormat::D24_UNORM_S8_UINT;

	auto desc = createDescription(&vs, &fs);
	desc.Blend.IndependentBlendEnable = true;
	desc.Blend.RenderTarget[0].BlendEnable = true;
	desc.Blend.RenderTarget[0].SrcBlend = Blend::SrcAlpha;
	desc.Blend.RenderTarget[0].DestBlend = Blend::InvSrcAlpha;
	desc.Blend.RenderTarget[1].RenderTargetWriteMask = ColourWriteEnable::Red | ColourWriteEnable::Green;
	desc.Rasterizer.CullMode = CullModeMethod::None;
	desc.Rasterizer.DepthBias = 2;
	desc.Rasterizer.SlopeScaledDepthBias = 1.5f;
	desc.DepthStencil.DepthFunc = ComparisonFunction::GreaterEqual;
	desc.DepthStencil.StencilEnable = true;
	desc.DepthStencil.FrontFace.StencilPassOp = StencilOperation::Replace;

	Runtime::Vulkan::GraphicsPipelineState pps{ _context.get(), &layout, desc, rt_layout };
	EXPECT_NE(static_cast<VkPipeline>(pps), VkPipeline{ VK_NULL_HANDLE });
}
