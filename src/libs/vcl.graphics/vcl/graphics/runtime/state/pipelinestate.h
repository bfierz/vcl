/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <vector>

// VCL
#include <vcl/graphics/runtime/resource/shader.h>
#include <vcl/graphics/runtime/state/blendstate.h>
#include <vcl/graphics/runtime/state/depthstencilstate.h>
#include <vcl/graphics/runtime/state/inputlayout.h>
#include <vcl/graphics/runtime/state/rasterizerstate.h>

namespace Vcl { namespace Graphics { namespace Runtime
{
	enum class PrimitiveType
	{
		Undefined = 0,
		Pointlist = 1,
		Linelist = 2,
		Linestrip = 3,
		Trianglelist = 4,
		Trianglestrip = 5,
		LinelistAdj = 10,
		LinestripAdj = 11,
		TrianglelistAdj = 12,
		TrianglestripAdj = 13,
		Patch = 14
	};

	struct InputAssemblyDescription
	{
		PrimitiveType Topology{ PrimitiveType::Undefined };
		bool PrimitiveRestartEnable{ false };
	};

	struct PipelineStateDescription
	{
		// Vertex shader
		Runtime::Shader* VertexShader{ nullptr };

		// Tessellation Control shader
		Runtime::Shader* TessControlShader{ nullptr };

		// Tessellation Evaluation shader
		Runtime::Shader* TessEvalShader{ nullptr };

		// Geometry shader
		Runtime::Shader* GeometryShader{ nullptr };

		// Fragment shader
		Runtime::Shader* FragmentShader{ nullptr };

		// Input assembly state
		Runtime::InputAssemblyDescription InputAssembly;

		// Input layout
		Runtime::InputLayoutDescription InputLayout;

		// Blend state
		Runtime::BlendDescription Blend;

		// Rasterizer state
		Runtime::RasterizerDescription Rasterizer;

		// Depth stencil state
		Runtime::DepthStencilDescription DepthStencil;

	};

	/*!
	 *	\brief Abstraction of the global states the render pipeline uses
	 */
	class PipelineState
	{
	public:
		virtual ~PipelineState() = default;
	};
}}}
