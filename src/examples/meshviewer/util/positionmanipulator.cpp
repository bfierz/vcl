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
#include "positionmanipulator.h"

// VCL
#include <vcl/geometry/meshfactory.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>

// Project specific
#include "shaderutils.h"

namespace
{
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> createBuffer(const void* buffer, size_t nr_elements, size_t stride)
	{
		using namespace Vcl::Graphics::Runtime;

		BufferDescription desc;
		desc.Usage = ResourceUsage::Default;
		desc.SizeInBytes = static_cast<uint32_t>(nr_elements * stride);

		BufferInitData data;
		data.Data = buffer;
		data.SizeInBytes = nr_elements * stride;

		return std::make_unique<Vcl::Graphics::Runtime::OpenGL::Buffer>(desc, false, false, &data);
	}
}

namespace Vcl { namespace Editor { namespace Util
{
	PositionManipulator::PositionManipulator()
	{
		using namespace Vcl::Geometry;
		using Vcl::Graphics::Runtime::OpenGL::PipelineState;
		using Vcl::Graphics::Runtime::OpenGL::Shader;
		using Vcl::Graphics::Runtime::Blend;
		using Vcl::Graphics::Runtime::InputLayoutDescription;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::VertexDataClassification;
		using Vcl::Graphics::SurfaceFormat;

		 auto mesh = TriMeshFactory::createArrow(0.05f, 0.1f, 0.8f, 0.2f, 10);

		 // Create the index buffer
		 _indexStride = sizeof(IndexDescriptionTrait<TriMesh>::Face);
		 _indices = createBuffer(mesh->faces()->data(), mesh->nrFaces(), _indexStride);

		 // Create the position buffer
		 _positionStride = sizeof(IndexDescriptionTrait<TriMesh>::Vertex);
		 _positions = createBuffer(mesh->vertices()->data(), mesh->nrVertices(), _positionStride);

		 // Create the normal buffer
		 auto normals = mesh->vertexProperty<Eigen::Vector3f>("Normals");
		 _normalStride = sizeof(Eigen::Vector3f);
		 _normals = createBuffer(normals->data(), normals->size(), _normalStride);

		 // Data layout
		 InputLayoutDescription opaque_layout =
		 {
			 { "Position",  SurfaceFormat::R32G32B32_FLOAT, 0, 0, 0, VertexDataClassification::VertexDataPerObject, 0 },
			 { "Normal",    SurfaceFormat::R32G32B32_FLOAT, 0, 1, 0, VertexDataClassification::VertexDataPerObject, 0 },
		 };

		 Shader arrow_colour_vert = createShader(ShaderType::VertexShader,   ":/shaders/debug/positionhandle_arrow.vert");
		 Shader arrow_colour_frag = createShader(ShaderType::FragmentShader, ":/shaders/debug/positionhandle_arrow.frag");

		 Shader square_colour_vert = createShader(ShaderType::VertexShader,   ":/shaders/debug/positionhandle_square.vert");
		 Shader square_colour_frag = createShader(ShaderType::FragmentShader, ":/shaders/debug/positionhandle_square.frag");

		 PipelineStateDescription opaque_colour_psdesc;
		 opaque_colour_psdesc.InputLayout = opaque_layout;
		 opaque_colour_psdesc.VertexShader = &arrow_colour_vert;
		 opaque_colour_psdesc.FragmentShader = &arrow_colour_frag;
		 _opaquePipelineState = make_owner<PipelineState>(opaque_colour_psdesc);

		 PipelineStateDescription transparent_colour_psdesc;
		 transparent_colour_psdesc.VertexShader = &square_colour_vert;
		 transparent_colour_psdesc.FragmentShader = &square_colour_frag;
		 transparent_colour_psdesc.Blend.RenderTarget[0].BlendEnable = true;
		 transparent_colour_psdesc.Blend.RenderTarget[0].SrcBlend = Blend::SrcAlpha;
		 transparent_colour_psdesc.Blend.RenderTarget[0].DestBlend = Blend::InvSrcAlpha;
		 _transparentPipelineState = make_owner<PipelineState>(transparent_colour_psdesc);
	}

	void PositionManipulator::draw(
		gsl::not_null<Vcl::Graphics::Runtime::GraphicsEngine*> engine,
		const Eigen::Matrix4f& T
	)
	{
		Require(_opaquePipelineState, "Opaque pipeline state is initialized.");
		Require(_transparentPipelineState, "Transparent pipeline state is initialized.");

		using BufferGL = Vcl::Graphics::Runtime::OpenGL::Buffer;
		using PipelineStateGL = Vcl::Graphics::Runtime::OpenGL::PipelineState;

		////////////////////////////////////////////////////////////////////////
		// Draw the arrows using instancing
		////////////////////////////////////////////////////////////////////////

		// Configure the layout
		engine->setPipelineState(_opaquePipelineState);

		// Set the shader constants
		static_cast<PipelineStateGL*>(_opaquePipelineState.get())->program().setUniform("ModelMatrix", T);

		// Bind the buffers
		glBindVertexBuffer(0, static_cast<BufferGL*>(_positions.get())->id(), 0, _positionStride);
		glBindVertexBuffer(1, static_cast<BufferGL*>(_normals.get())->id(),   0, _normalStride);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, static_cast<BufferGL*>(_indices.get())->id());

		// Render the mesh
		glDrawElementsInstanced(GL_TRIANGLES, static_cast<uint32_t>(_indices->sizeInBytes()) / sizeof(uint32_t), GL_UNSIGNED_INT, nullptr, 3);

		////////////////////////////////////////////////////////////////////////
		// Draw the squares
		////////////////////////////////////////////////////////////////////////

		// Configure the layout
		engine->setPipelineState(_transparentPipelineState);

		// Set the shader constants
		static_cast<PipelineStateGL*>(_transparentPipelineState.get())->program().setUniform("ModelMatrix", T);

		// Render the mesh
		glDrawArrays(GL_TRIANGLES, 0, 18);

	}
}}}
