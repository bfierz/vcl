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
#include <vcl/graphics/runtime/state/devicefeatures.h>

namespace Vcl { namespace Graphics { namespace Runtime {
	namespace {
		bool isDualSourceFactor(Blend factor)
		{
			return factor == Blend::Src1Colour || factor == Blend::InvSrc1Colour || factor == Blend::Src1Alpha || factor == Blend::InvSrc1Alpha;
		}

		bool usesDualSourceBlend(const BlendDescription& desc, size_t nr_render_targets)
		{
			// Without independent blending the first state applies to all render-targets
			const size_t nr_states = desc.IndependentBlendEnable ? nr_render_targets : 1;
			for (size_t i = 0; i < nr_states && i < desc.RenderTarget.size(); i++)
			{
				const auto& rt = desc.RenderTarget[i];
				if (rt.BlendEnable && (isDualSourceFactor(rt.SrcBlend) || isDualSourceFactor(rt.DestBlend) || isDualSourceFactor(rt.SrcBlendAlpha) || isDualSourceFactor(rt.DestBlendAlpha)))
					return true;
			}
			return false;
		}

		//! Checks shared by all pipelines rasterizing primitives
		std::string validateRasterization(
			const BlendDescription& blend,
			const RasterizerDescription& rasterizer,
			const RenderTargetLayout& render_targets,
			const DeviceFeatures& features)
		{
			if (render_targets.ColourFormats.size() > features.MaxRenderTargets)
				return "Number of render-targets is not supported.";
			if (render_targets.ColourFormats.size() > blend.RenderTarget.size())
				return "Number of render-targets exceeds the number of blend states.";
			if (blend.IndependentBlendEnable && !features.IndependentBlend)
				return "Independent blending is not supported.";
			if (blend.LogicOpEnable && !features.LogicOp)
				return "Logic operations are not supported.";
			if (usesDualSourceBlend(blend, render_targets.ColourFormats.size()) && !features.DualSourceBlend)
				return "Dual-source blending is not supported.";
			if (rasterizer.FillMode == FillModeMethod::Wireframe && !features.WireframeFill)
				return "Wireframe fill mode is not supported.";
			if (rasterizer.DepthBiasClamp != 0.0f && !features.DepthBiasClamp)
				return "Depth-bias clamping is not supported.";
			if (!rasterizer.DepthClipEnable && !features.DepthClipControl)
				return "Disabling depth clipping is not supported.";

			return {};
		}
	}

	std::string validate(const PipelineStateDescription& desc, const DeviceFeatures& features)
	{
		if (!desc.VertexShader)
			return "Vertex shader is not set.";
		if (desc.GeometryShader && !features.GeometryShaders)
			return "Geometry shaders are not supported.";
		if ((desc.TessControlShader || desc.TessEvalShader) && !features.TessellationShaders)
			return "Tessellation shaders are not supported.";
		if ((desc.TessControlShader == nullptr) != (desc.TessEvalShader == nullptr))
			return "Tessellation requires a control and an evaluation shader.";
		if ((desc.InputAssembly.Topology == PrimitiveType::Patch) != (desc.TessEvalShader != nullptr))
			return "Patches are used with tessellation shaders, and only with them.";

		return validateRasterization(desc.Blend, desc.Rasterizer, desc.RenderTargets, features);
	}

	std::string validate(const MeshPipelineStateDescription& desc, const DeviceFeatures& features)
	{
		if (!features.MeshShaders)
			return "Mesh shaders are not supported.";
		if (!desc.MeshShader)
			return "Mesh shader is not set.";
		if (desc.TaskShader && !features.TaskShaders)
			return "Task shaders are not supported.";

		return validateRasterization(desc.Blend, desc.Rasterizer, desc.RenderTargets, features);
	}

	std::string validate(const BindlessLayoutDescription& desc, const DeviceFeatures& features)
	{
		if (!features.BindlessResources)
			return "Bindless resources are not supported.";
		if (desc.ResourceCapacity == 0 || desc.SamplerCapacity == 0)
			return "Heaps are empty.";
		if (desc.ResourceCapacity > features.MaxResourceHeapCapacity)
			return "Resource heap capacity is not supported.";
		if (desc.SamplerCapacity > features.MaxSamplerHeapCapacity)
			return "Sampler heap capacity is not supported.";
		if (desc.PushConstantSize % 4 != 0)
			return "Push-constant size is not a multiple of 4.";
		if (desc.PushConstantSize > features.MaxPushConstantSize)
			return "Push-constant size is not supported.";

		for (size_t i = 0; i < desc.StaticSamplers.size(); i++)
		{
			for (size_t j = i + 1; j < desc.StaticSamplers.size(); j++)
			{
				if (desc.StaticSamplers[i].Register == desc.StaticSamplers[j].Register)
					return "Static samplers share a register.";
			}
		}

		return {};
	}
}}}
