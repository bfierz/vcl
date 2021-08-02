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

namespace {
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> createBuffer(const void* buffer, size_t nr_elements, size_t stride)
	{
		using namespace Vcl::Graphics::Runtime;

		BufferDescription desc;
		desc.Usage = BufferUsage::Vertex | BufferUsage::Index;
		desc.SizeInBytes = static_cast<uint32_t>(nr_elements * stride);

		BufferInitData data;
		data.Data = buffer;
		data.SizeInBytes = nr_elements * stride;

		return std::make_unique<Vcl::Graphics::Runtime::OpenGL::Buffer>(desc, &data);
	}
}

namespace Vcl { namespace Editor { namespace Util {
	PositionManipulator::PositionManipulator()
	{
		using namespace Vcl::Geometry;
		using Vcl::Graphics::SurfaceFormat;
		using Vcl::Graphics::Runtime::Blend;
		using Vcl::Graphics::Runtime::InputLayoutDescription;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::VertexDataClassification;
		using Vcl::Graphics::Runtime::OpenGL::PipelineState;
		using Vcl::Graphics::Runtime::OpenGL::Shader;

		/////////////////////////////////////////////////////////////////////////
		// Arrow used as translation handle
		/////////////////////////////////////////////////////////////////////////
		auto arrow_mesh = TriMeshFactory::createArrow(0.05f, 0.1f, 0.8f, 0.2f, 10);

		// Create the index buffer
		_arrow.indexStride = sizeof(IndexDescriptionTrait<TriMesh>::Face);
		_arrow.indices = createBuffer(arrow_mesh->faces()->data(), arrow_mesh->nrFaces(), _arrow.indexStride);

		// Create the position buffer
		_arrow.positionStride = sizeof(IndexDescriptionTrait<TriMesh>::Vertex);
		_arrow.positions = createBuffer(arrow_mesh->vertices()->data(), arrow_mesh->nrVertices(), _arrow.positionStride);

		// Create the normal buffer
		auto arrow_normals = arrow_mesh->vertexProperty<Eigen::Vector3f>("Normals");
		_arrow.normalStride = sizeof(Eigen::Vector3f);
		_arrow.normals = createBuffer(arrow_normals->data(), arrow_normals->size(), _arrow.normalStride);

		/////////////////////////////////////////////////////////////////////////
		// Torus used as rotation handles
		/////////////////////////////////////////////////////////////////////////
		auto torus_mesh = TriMeshFactory::createArrow(0.05f, 0.1f, 0.8f, 0.2f, 10);

		// Create the index buffer
		_torus.indexStride = sizeof(IndexDescriptionTrait<TriMesh>::Face);
		_torus.indices = createBuffer(torus_mesh->faces()->data(), torus_mesh->nrFaces(), _torus.indexStride);

		// Create the position buffer
		_torus.positionStride = sizeof(IndexDescriptionTrait<TriMesh>::Vertex);
		_torus.positions = createBuffer(torus_mesh->vertices()->data(), torus_mesh->nrVertices(), _torus.positionStride);

		// Create the normal buffer
		auto torus_normals = torus_mesh->vertexProperty<Eigen::Vector3f>("Normals");
		_torus.normalStride = sizeof(Eigen::Vector3f);
		_torus.normals = createBuffer(torus_normals->data(), torus_normals->size(), _torus.normalStride);

		/////////////////////////////////////////////////////////////////////////
		// Pipeline states
		/////////////////////////////////////////////////////////////////////////
		InputLayoutDescription opaque_layout = {
			{ { 0, sizeof(Eigen::Vector3f), Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerObject },
			  { 1, sizeof(Eigen::Vector4f), Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerObject } },
			{ { "Position", SurfaceFormat::R32G32B32_FLOAT, 0, 0, 0 },
			  { "Normal", SurfaceFormat::R32G32B32_FLOAT, 0, 1, 0 } }
		};

		Shader arrow_colour_vert = createShader(ShaderType::VertexShader, ":/shaders/debug/positionhandle_arrow.vert");
		Shader arrow_colour_frag = createShader(ShaderType::FragmentShader, ":/shaders/debug/positionhandle_arrow.frag");

		Shader square_colour_vert = createShader(ShaderType::VertexShader, ":/shaders/debug/positionhandle_square.vert");
		Shader square_colour_frag = createShader(ShaderType::FragmentShader, ":/shaders/debug/positionhandle_square.frag");

		Shader arrow_id_vert = createShader(ShaderType::VertexShader, ":/shaders/debug/positionhandle_arrow_id.vert");
		Shader square_id_vert = createShader(ShaderType::VertexShader, ":/shaders/debug/positionhandle_square_id.vert");

		Shader id_frag = createShader(ShaderType::FragmentShader, ":/shaders/objectid.frag");

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
		transparent_colour_psdesc.Rasterizer.CullMode = Vcl::Graphics::Runtime::CullModeMethod::None;
		_transparentPipelineState = make_owner<PipelineState>(transparent_colour_psdesc);

		PipelineStateDescription opaque_id_psdesc;
		opaque_id_psdesc.InputLayout = opaque_layout;
		opaque_id_psdesc.VertexShader = &arrow_id_vert;
		opaque_id_psdesc.FragmentShader = &id_frag;
		_opaqueIdPipelineState = make_owner<PipelineState>(opaque_id_psdesc);

		PipelineStateDescription transaparent_id_psdesc;
		transaparent_id_psdesc.VertexShader = &square_id_vert;
		transaparent_id_psdesc.FragmentShader = &id_frag;
		transaparent_id_psdesc.Rasterizer.CullMode = Vcl::Graphics::Runtime::CullModeMethod::None;
		_transparentIdPipelineState = make_owner<PipelineState>(transaparent_id_psdesc);
	}

	void PositionManipulator::drawIds(
		Vcl::ref_ptr<Vcl::Graphics::Runtime::GraphicsEngine> engine,
		unsigned int id,
		const Eigen::Matrix4f& T
	)
	{
		VclRequire(_opaqueIdPipelineState, "Opaque pipeline state is initialized.");
		VclRequire(_transparentIdPipelineState, "Transparent pipeline state is initialized.");

		using PipelineStateGL = Vcl::Graphics::Runtime::OpenGL::PipelineState;

		static_cast<PipelineStateGL*>(_opaqueIdPipelineState.get())->program().setUniform<int>("ObjectIdx", static_cast<int>(id));
		static_cast<PipelineStateGL*>(_transparentIdPipelineState.get())->program().setUniform<int>("ObjectIdx", static_cast<int>(id));

		draw(engine, T, _opaqueIdPipelineState, _transparentIdPipelineState);
	}

	void PositionManipulator::draw(
		Vcl::ref_ptr<Vcl::Graphics::Runtime::GraphicsEngine> engine,
		const Eigen::Matrix4f& T
	)
	{
		VclRequire(_opaquePipelineState, "Opaque pipeline state is initialized.");
		VclRequire(_transparentPipelineState, "Transparent pipeline state is initialized.");

		draw(engine, T, _opaquePipelineState, _transparentPipelineState);
	}

	void PositionManipulator::draw(
		Vcl::ref_ptr<Vcl::Graphics::Runtime::GraphicsEngine> engine,
		const Eigen::Matrix4f& T,
		ref_ptr<Vcl::Graphics::Runtime::PipelineState> opaque_ps,
		ref_ptr<Vcl::Graphics::Runtime::PipelineState> transparent_ps
	)
	{
		VclRequire(opaque_ps, "Opaque pipeline state is initialized.");
		VclRequire(transparent_ps, "Transparent pipeline state is initialized.");

		using BufferGL = Vcl::Graphics::Runtime::OpenGL::Buffer;
		using PipelineStateGL = Vcl::Graphics::Runtime::OpenGL::PipelineState;

		////////////////////////////////////////////////////////////////////////
		// Draw the arrows using instancing
		////////////////////////////////////////////////////////////////////////

		// Configure the layout
		engine->setPipelineState(opaque_ps);

		// Set the shader constants
		static_cast<PipelineStateGL*>(opaque_ps.get())->program().setUniform("ModelMatrix", T);

		// Bind the buffers
		glBindVertexBuffer(0, static_cast<BufferGL*>(_arrow.positions.get())->id(), 0, _arrow.positionStride);
		glBindVertexBuffer(1, static_cast<BufferGL*>(_arrow.normals.get())->id(), 0, _arrow.normalStride);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, static_cast<BufferGL*>(_arrow.indices.get())->id());

		// Render the mesh
		glDrawElementsInstanced(GL_TRIANGLES, static_cast<uint32_t>(_arrow.indices->sizeInBytes()) / sizeof(uint32_t), GL_UNSIGNED_INT, nullptr, 3);

		////////////////////////////////////////////////////////////////////////
		// Draw the squares
		////////////////////////////////////////////////////////////////////////

		// Configure the layout
		engine->setPipelineState(transparent_ps);

		// Set the shader constants
		static_cast<PipelineStateGL*>(transparent_ps.get())->program().setUniform("ModelMatrix", T);

		// Render the mesh
		glDrawArrays(GL_TRIANGLES, 0, 18);
	}
}}}
