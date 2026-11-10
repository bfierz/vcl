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
#include <vcl/graphics/runtime/state/devicefeatures.h>
#include <vcl/graphics/vulkan/context.h>

// Test
#include "test.h"

namespace {
	using namespace Vcl::Graphics::Runtime;

	//! Shader without backend, used to mark a stage as set
	class StageMarker : public Shader
	{
	public:
		StageMarker(ShaderType type)
		: Shader(type, 0) {}
	};

	DeviceFeatures makeFeatures()
	{
		DeviceFeatures features;
		features.BindlessResources = true;
		features.MaxRenderTargets = 8;
		features.MaxPushConstantSize = 128;
		features.MaxResourceHeapCapacity = 1000000;
		features.MaxSamplerHeapCapacity = 2048;
		return features;
	}

	PipelineStateDescription makePipeline(Shader* vs)
	{
		PipelineStateDescription desc;
		desc.VertexShader = vs;
		desc.InputAssembly.Topology = PrimitiveType::Trianglelist;
		desc.RenderTargets.ColourFormats = { Vcl::Graphics::SurfaceFormat::R8G8B8A8_UNORM };
		return desc;
	}
}

class VulkanDeviceFeaturesTest : public VulkanTest
{
};

TEST(DeviceFeatures, PipelineUsesOnlyAvailableFeatures)
{
	StageMarker vs{ ShaderType::VertexShader };
	StageMarker gs{ ShaderType::GeometryShader };
	const auto features = makeFeatures();

	EXPECT_EQ(validate(makePipeline(&vs), features), "");
	EXPECT_NE(validate(makePipeline(nullptr), features), "");

	auto geometry = makePipeline(&vs);
	geometry.GeometryShader = &gs;
	EXPECT_NE(validate(geometry, features), "");

	auto wireframe = makePipeline(&vs);
	wireframe.Rasterizer.FillMode = FillModeMethod::Wireframe;
	EXPECT_NE(validate(wireframe, features), "");

	auto no_depth_clip = makePipeline(&vs);
	no_depth_clip.Rasterizer.DepthClipEnable = false;
	EXPECT_NE(validate(no_depth_clip, features), "");

	auto independent_blend = makePipeline(&vs);
	independent_blend.Blend.IndependentBlendEnable = true;
	EXPECT_NE(validate(independent_blend, features), "");

	auto dual_source = makePipeline(&vs);
	dual_source.Blend.RenderTarget[0].BlendEnable = true;
	dual_source.Blend.RenderTarget[0].DestBlend = Blend::InvSrc1Colour;
	EXPECT_NE(validate(dual_source, features), "");

	auto too_many_targets = makePipeline(&vs);
	too_many_targets.RenderTargets.ColourFormats.resize(9, Vcl::Graphics::SurfaceFormat::R8G8B8A8_UNORM);
	EXPECT_NE(validate(too_many_targets, features), "");

	// The same pipelines are valid once the device provides the features
	auto all_features = features;
	all_features.GeometryShaders = true;
	all_features.WireframeFill = true;
	all_features.DepthClipControl = true;
	all_features.IndependentBlend = true;
	all_features.DualSourceBlend = true;
	EXPECT_EQ(validate(geometry, all_features), "");
	EXPECT_EQ(validate(wireframe, all_features), "");
	EXPECT_EQ(validate(no_depth_clip, all_features), "");
	EXPECT_EQ(validate(independent_blend, all_features), "");
	EXPECT_EQ(validate(dual_source, all_features), "");
}

TEST(DeviceFeatures, TessellationNeedsBothStagesAndPatches)
{
	StageMarker vs{ ShaderType::VertexShader };
	StageMarker cs{ ShaderType::ControlShader };
	StageMarker es{ ShaderType::EvaluationShader };
	auto features = makeFeatures();
	features.TessellationShaders = true;

	auto desc = makePipeline(&vs);
	desc.TessControlShader = &cs;
	desc.TessEvalShader = &es;
	EXPECT_NE(validate(desc, features), "") << "Triangle list with tessellation";

	desc.InputAssembly.Topology = PrimitiveType::Patch;
	EXPECT_EQ(validate(desc, features), "");

	desc.TessControlShader = nullptr;
	EXPECT_NE(validate(desc, features), "") << "Missing control shader";

	features.TessellationShaders = false;
	desc.TessControlShader = &cs;
	EXPECT_NE(validate(desc, features), "");
}

TEST(DeviceFeatures, MeshPipeline)
{
	StageMarker ms{ ShaderType::MeshShader };
	StageMarker ts{ ShaderType::TaskShader };
	auto features = makeFeatures();

	MeshPipelineStateDescription desc;
	desc.MeshShader = &ms;
	EXPECT_NE(validate(desc, features), "");

	features.MeshShaders = true;
	EXPECT_EQ(validate(desc, features), "");

	desc.TaskShader = &ts;
	EXPECT_NE(validate(desc, features), "");

	features.TaskShaders = true;
	EXPECT_EQ(validate(desc, features), "");
}

TEST(DeviceFeatures, BindlessLayout)
{
	const auto features = makeFeatures();

	EXPECT_EQ(validate(BindlessLayoutDescription{}, features), "");

	BindlessLayoutDescription desc;
	desc.PushConstantSize = 256;
	EXPECT_NE(validate(desc, features), "");
	desc.PushConstantSize = 30;
	EXPECT_NE(validate(desc, features), "");

	desc = {};
	desc.SamplerCapacity = 4096;
	EXPECT_NE(validate(desc, features), "");

	desc = {};
	desc.ResourceCapacity = 0;
	EXPECT_NE(validate(desc, features), "");

	desc = {};
	desc.StaticSamplers = { { 2, {} }, { 2, {} } };
	EXPECT_NE(validate(desc, features), "");

	auto no_bindless = features;
	no_bindless.BindlessResources = false;
	EXPECT_NE(validate(BindlessLayoutDescription{}, no_bindless), "");
}

TEST_F(VulkanDeviceFeaturesTest, ContextReportsFeatures)
{
	const auto& features = _context->features();

	EXPECT_TRUE(features.BindlessResources);
	EXPECT_EQ(features.MeshShaders, _context->supportsMeshShaders());
	EXPECT_GE(features.MaxRenderTargets, 4u);
	EXPECT_GE(features.MaxPushConstantSize, 128u);

	// The default layout has to work on every supported device
	EXPECT_EQ(validate(BindlessLayoutDescription{}, features), "");
}
