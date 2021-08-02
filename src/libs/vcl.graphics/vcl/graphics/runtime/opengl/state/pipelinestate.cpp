/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL {
	PipelineState::PipelineState(const PipelineStateDescription& desc)
	: _inputLayout(desc.InputLayout)
	, _inputAssembly(desc.InputAssembly)
	, _blendState(desc.Blend)
	, _depthStencilState(desc.DepthStencil)
	, _rasterizerState(desc.Rasterizer)
	{
		ShaderProgramDescription shader_desc;
		shader_desc.InputLayout = desc.InputLayout;
		shader_desc.VertexShader = static_cast<Runtime::OpenGL::Shader*>(desc.VertexShader);
		shader_desc.TessControlShader = static_cast<Runtime::OpenGL::Shader*>(desc.TessControlShader);
		shader_desc.TessEvalShader = static_cast<Runtime::OpenGL::Shader*>(desc.TessEvalShader);
		shader_desc.GeometryShader = static_cast<Runtime::OpenGL::Shader*>(desc.GeometryShader);
		shader_desc.FragmentShader = static_cast<Runtime::OpenGL::Shader*>(desc.FragmentShader);

		_shaderProgram = std::make_unique<ShaderProgram>(shader_desc);
	}

	void PipelineState::bind()
	{
		// Bind the shader
		_shaderProgram->bind();

		// Bind the input description
		_inputLayout.bind();

		// Bind the rasterizer configueration
		_rasterizerState.bind();

		// Bind the depth-stencil configuration
		_depthStencilState.bind();

		// Bind the blending configuration
		_blendState.bind();
	}
}}}}
