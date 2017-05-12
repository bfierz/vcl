/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#include "rendertargetdebugger.h"

// VCL
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/state/sampler.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>

// Project specific
#include "shaderutils.h"

namespace Vcl { namespace Editor { namespace Util
{
	RendertargetDebugger::RendertargetDebugger()
	{
		//using namespace Vcl::Geometry;
		using Vcl::Graphics::Runtime::OpenGL::PipelineState;
		using Vcl::Graphics::Runtime::OpenGL::Sampler;
		using Vcl::Graphics::Runtime::OpenGL::Shader;
		//using Vcl::Graphics::Runtime::InputLayoutDescription;
		using Vcl::Graphics::Runtime::FilterType;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::SamplerDescription;
		using Vcl::Graphics::Runtime::ShaderType;
		//using Vcl::Graphics::Runtime::VertexDataClassification;
		//using Vcl::Graphics::SurfaceFormat;

		 /////////////////////////////////////////////////////////////////////////
		 // Pipeline states
		 /////////////////////////////////////////////////////////////////////////
		
		 Shader present_vert = createShader(ShaderType::VertexShader,   ":/shaders/debug/renderttarget_present.vert");
		 Shader present_frag = createShader(ShaderType::FragmentShader, ":/shaders/debug/renderttarget_present.frag");

		 PipelineStateDescription present_psdesc;
		 present_psdesc.VertexShader = &present_vert;
		 present_psdesc.FragmentShader = &present_frag;
		 _presentationPipelineState = make_owner<PipelineState>(present_psdesc);

		 // Create a nearest neighbour sampler for the render-target visualization
		 SamplerDescription sampler_desc;
		 sampler_desc.Filter = FilterType::MinMagMipPoint;
		 _rtSampler = std::make_unique<Sampler>(sampler_desc);
	}

	void RendertargetDebugger::draw(
		gsl::not_null<Vcl::Graphics::Runtime::GraphicsEngine*> engine,
		const Vcl::Graphics::Runtime::Texture& texture,
		const Eigen::Vector4f& loc_size
	)
	{
		using BufferGL = Vcl::Graphics::Runtime::OpenGL::Buffer;
		using PipelineStateGL = Vcl::Graphics::Runtime::OpenGL::PipelineState;

		////////////////////////////////////////////////////////////////////////
		// Draw the texture on the screen
		////////////////////////////////////////////////////////////////////////

		// Configure the layout
		engine->setPipelineState(_presentationPipelineState);

		// Set the shader constants
		static_cast<PipelineStateGL*>(_presentationPipelineState.get())->program().setUniform("Viewport", loc_size);
		const auto tex_handle = static_cast<PipelineStateGL*>(_presentationPipelineState.get())->program().uniform("inputTex");
		static_cast<PipelineStateGL*>(_presentationPipelineState.get())->program().setTexture(tex_handle, &texture, _rtSampler.get());

		// Render the mesh
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}
}}}
