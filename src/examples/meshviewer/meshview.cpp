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
#include "meshview.h"

// Qt
#include <QtQuick/QQuickWindow>

// VCL
#include <vcl/graphics/runtime/opengl/resource/shader.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>
#include <vcl/graphics/runtime/opengl/graphicsengine.h>

#include "scene.h"

namespace
{
	Vcl::Graphics::Runtime::OpenGL::Shader createShader(Vcl::Graphics::Runtime::ShaderType type, QString path)
	{
		QFile shader_file{ path };
		shader_file.open(QIODevice::ReadOnly);
		Check(shader_file.isOpen(), "Shader file is open.");

		return{ type, 0, shader_file.readAll().data() };
	}
}

namespace
{
	struct PerFrameCameraData
	{
		// Viewport (x, y, w, h)
		Eigen::Vector4f Viewport;

		// Transform from world to view space
		Eigen::Matrix4f ViewMatrix;

		// Transform from view to screen space
		Eigen::Matrix4f ProjectionMatrix;
	};
}

FboRenderer::FboRenderer()
{
	using Vcl::Graphics::Runtime::OpenGL::InputLayout;
	using Vcl::Graphics::Runtime::OpenGL::PipelineState;
	using Vcl::Graphics::Runtime::OpenGL::Shader;
	using Vcl::Graphics::Runtime::OpenGL::ShaderProgramDescription;
	using Vcl::Graphics::Runtime::OpenGL::ShaderProgram;
	using Vcl::Graphics::Runtime::InputLayoutDescription;
	using Vcl::Graphics::Runtime::PipelineStateDescription;
	using Vcl::Graphics::Runtime::ShaderType;
	using Vcl::Graphics::Runtime::VertexDataClassification;
	using Vcl::Graphics::SurfaceFormat;

	_engine = std::make_unique<Vcl::Graphics::Runtime::OpenGL::GraphicsEngine>();

	InputLayoutDescription opaqueTriLayout =
	{
		{ "Index",  SurfaceFormat::R32G32B32_FLOAT, 0, 0, 0, VertexDataClassification::VertexDataPerObject, 0 },
		{ "Colour", SurfaceFormat::R32G32B32A32_FLOAT, 0, 1, 0, VertexDataClassification::VertexDataPerObject, 0 },
	};
	_opaqueTriLayout = std::make_unique<InputLayout>(opaqueTriLayout);
	
	InputLayoutDescription idTetraLayout =
	{
		{ "Index",  SurfaceFormat::R32G32B32A32_SINT, 0, 0, 0, VertexDataClassification::VertexDataPerObject, 0 },
	};
	_idTetraLayout = std::make_unique<InputLayout>(idTetraLayout);

	InputLayoutDescription opaqueTetraLayout =
	{
		{ "Index",  SurfaceFormat::R32G32B32A32_SINT, 0, 0, 0, VertexDataClassification::VertexDataPerObject, 0 },
		{ "Colour", SurfaceFormat::R32G32B32A32_FLOAT, 0, 1, 0, VertexDataClassification::VertexDataPerObject, 0 },
	};
	_opaqueTetraLayout = std::make_unique<InputLayout>(opaqueTetraLayout);

	Shader opaqueTriVert = createShader(ShaderType::VertexShader, ":/shaders/trimesh.vert");
	Shader opaqueTriGeom = createShader(ShaderType::GeometryShader, ":/shaders/trimesh.geom");

	Shader opaqueTetraVert = createShader(ShaderType::VertexShader, ":/shaders/tetramesh.vert");
	Shader opaqueTetraGeom = createShader(ShaderType::GeometryShader, ":/shaders/tetramesh.geom");
	Shader idTetraVert = createShader(ShaderType::VertexShader, ":/shaders/objectid_tetramesh.vert");
	Shader idTetraGeom = createShader(ShaderType::GeometryShader, ":/shaders/objectid_tetramesh.geom");

	Shader meshFrag = createShader(ShaderType::FragmentShader, ":/shaders/mesh.frag");
	Shader idFrag = createShader(ShaderType::FragmentShader, ":/shaders/objectid.frag");

	ShaderProgramDescription opaqueTriDesc;
	opaqueTriDesc.InputLayout = opaqueTriLayout;
	opaqueTriDesc.VertexShader = &opaqueTriVert;
	opaqueTriDesc.GeometryShader = &opaqueTriGeom;
	opaqueTriDesc.FragmentShader = &meshFrag;
	_opaqueTriMeshShader = std::make_unique<ShaderProgram>(opaqueTriDesc);

	ShaderProgramDescription idTetraDesc;
	idTetraDesc.InputLayout = idTetraLayout;
	idTetraDesc.VertexShader = &idTetraVert;
	idTetraDesc.GeometryShader = &idTetraGeom;
	idTetraDesc.FragmentShader = &idFrag;
	_idTetraMeshShader = std::make_unique<ShaderProgram>(idTetraDesc);

	ShaderProgramDescription opaqueTetraDesc;
	opaqueTetraDesc.InputLayout = opaqueTetraLayout;
	opaqueTetraDesc.VertexShader = &opaqueTetraVert;
	opaqueTetraDesc.GeometryShader = &opaqueTetraGeom;
	opaqueTetraDesc.FragmentShader = &meshFrag;
	_opaqueTetraMeshShader = std::make_unique<ShaderProgram>(opaqueTetraDesc);

	PipelineStateDescription opaqueTetraPSDesc;
	opaqueTetraPSDesc.InputLayout = opaqueTetraLayout;
	opaqueTetraPSDesc.VertexShader = &opaqueTetraVert;
	opaqueTetraPSDesc.GeometryShader = &opaqueTetraGeom;
	opaqueTetraPSDesc.FragmentShader = &meshFrag;
	_opaqueTetraPipelineState = Vcl::make_owner<PipelineState>(opaqueTetraPSDesc);
}

void FboRenderer::render()
{
	_engine->beginFrame();

	glClearColor(0, 0, 0, 1);
	glClearDepth(1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (_owner)
	{
		auto scene = _owner->scene();

		////////////////////////////////////////////////////////////////////////
		// Prepare the environment
		////////////////////////////////////////////////////////////////////////
		Eigen::Matrix4f M = scene->modelMatrix();
		Eigen::Matrix4f V = scene->viewMatrix();
		Eigen::Matrix4f P = scene->projMatrix();
		
		auto cbuffer_cam = _engine->requestPerFrameConstantBuffer(sizeof(PerFrameCameraData));
		auto cbuffer_cam_ptr = reinterpret_cast<PerFrameCameraData*>(cbuffer_cam.data());
		cbuffer_cam_ptr->Viewport = { 0, 0, (float)_owner->width(), (float)_owner->height() };
		cbuffer_cam_ptr->ViewMatrix = V;
		cbuffer_cam_ptr->ProjectionMatrix = P;

		_opaqueTriMeshShader->setConstantBuffer("PerFrameCameraData", &cbuffer_cam.owner(), cbuffer_cam.offset(), cbuffer_cam.size());
		_opaqueTriMeshShader->setUniform(_opaqueTriMeshShader->uniform("ModelMatrix"), M);
		_idTetraMeshShader->setConstantBuffer("PerFrameCameraData", &cbuffer_cam.owner(), cbuffer_cam.offset(), cbuffer_cam.size());
		_idTetraMeshShader->setUniform(_idTetraMeshShader->uniform("ModelMatrix"), M);
		_opaqueTetraMeshShader->setConstantBuffer("PerFrameCameraData", &cbuffer_cam.owner(), cbuffer_cam.offset(), cbuffer_cam.size());
		_opaqueTetraMeshShader->setUniform(_opaqueTetraMeshShader->uniform("ModelMatrix"), M);

		auto surfaceMesh = scene->surfaceMesh();
		if (surfaceMesh)
		{
			// Configure the layout
			_opaqueTriMeshShader->bind();
			_opaqueTriLayout->bind();

			////////////////////////////////////////////////////////////////////
			// Render the mesh
			////////////////////////////////////////////////////////////////////

			// Set the vertex positions
			_opaqueTriMeshShader->setBuffer("VertexPositions", surfaceMesh->positions());

			// Bind the buffers
			glBindVertexBuffer(0, surfaceMesh->indices()->id(),     0, sizeof(Eigen::Vector3i));
			glBindVertexBuffer(1, surfaceMesh->faceColours()->id(), 0, sizeof(Eigen::Vector4f));

			// Render the mesh
			glDrawArrays(GL_POINTS, 0, surfaceMesh->nrFaces());
		}

		auto volumeMesh = scene->volumeMesh();
		if (volumeMesh)
		{
			// Configure the state
			_engine->setPipelineState(_opaqueTetraPipelineState);

			////////////////////////////////////////////////////////////////////
			// Render the mesh
			////////////////////////////////////////////////////////////////////

			// Set the vertex positions
			_opaqueTetraMeshShader->setBuffer("VertexPositions", volumeMesh->positions());

			// Bind the buffers
			glBindVertexBuffer(0, volumeMesh->indices()->id(),       0, sizeof(Eigen::Vector4i));
			glBindVertexBuffer(1, volumeMesh->volumeColours()->id(), 0, sizeof(Eigen::Vector4f));

			// Render the mesh
			glDrawArrays(GL_POINTS, 0, volumeMesh->nrVolumes());
		}

	}

	_engine->endFrame();
	_owner->window()->resetOpenGLState();
	update();
}

void FboRenderer::synchronize(QQuickFramebufferObject* item)
{
	auto* view = dynamic_cast<MeshView*>(item);
	if (view)
	{
		_owner = view;
	}

	if (_owner && _owner->scene())
	{
		_owner->scene()->update();
	}
}

QOpenGLFramebufferObject* FboRenderer::createFramebufferObject(const QSize &size)
{
	QOpenGLFramebufferObjectFormat format;
	format.setAttachment(QOpenGLFramebufferObject::Depth);
	format.setSamples(4);
	return new QOpenGLFramebufferObject(size, format);
}


MeshView::Renderer* MeshView::createRenderer() const
{
	return new FboRenderer();
}

void MeshView::geometryChanged(const QRectF & newGeometry, const QRectF & oldGeometry)
{
	QQuickFramebufferObject::geometryChanged(newGeometry, oldGeometry);

	if (scene())
	{
		scene()->camera()->setViewport(newGeometry.width(), newGeometry.height());
	}
}
