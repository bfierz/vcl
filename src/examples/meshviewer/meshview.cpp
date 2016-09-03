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
#include <QtCore/QRegularExpression>
#include <QtCore/QStringBuilder>
#include <QtQuick/QQuickWindow>

// VCL
#include <vcl/geometry/MarchingCubesTables.h>
#include <vcl/graphics/runtime/opengl/resource/shader.h>
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>
#include <vcl/graphics/runtime/opengl/graphicsengine.h>

// Application
namespace
{
#include "shaders/3DSceneBindings.h"
#include "shaders/MarchingCubes.h"
}

#include "util/frustumhelpers.h"
#include "scene.h"

namespace
{
	QString resolveShaderFile(QString full_path)
	{
		QRegularExpression dir_regex{ R"((.+/)(.+))" };
		QRegularExpressionMatch match;
		full_path.indexOf(dir_regex, 0, &match);
		Check(match.hasMatch(), "Split is successfull.");

		QString dir = match.captured(1);
		QString path = match.captured(2);

		QFile shader_file{ dir + path };
		shader_file.open(QIODevice::ReadOnly | QIODevice::Text);
		Check(shader_file.isOpen(), "Shader file is open.");

		// Resolve include files (only one level supported atm)
		QString builder;
		QTextStream textStream(&shader_file);

		QRegularExpression inc_regex{ R"(#.*include.*[<"](.+)[">])" };
		while (!textStream.atEnd())
		{
			auto curr_tok = textStream.readLine();

			QRegularExpressionMatch match;
			if (curr_tok.indexOf(inc_regex, 0, &match) >= 0 && match.hasMatch())
			{
				QString included_file = resolveShaderFile(dir + match.captured(1));
				builder = builder % included_file % "\n";
			}
			else if (curr_tok.indexOf("GL_GOOGLE_include_directive") >= 0)
			{
				continue;
			}
			else
			{
				builder = builder % curr_tok % "\n";
			}
		}

		shader_file.close();

		return builder;
	}

	Vcl::Graphics::Runtime::OpenGL::Shader createShader(Vcl::Graphics::Runtime::ShaderType type, QString path)
	{
		QString data = resolveShaderFile(path);

		return{ type, 0, data.toUtf8().data() };
	}
}

FboRenderer::FboRenderer()
{
	using Vcl::Graphics::Runtime::OpenGL::Buffer;
	using Vcl::Graphics::Runtime::OpenGL::InputLayout;
	using Vcl::Graphics::Runtime::OpenGL::PipelineState;
	using Vcl::Graphics::Runtime::OpenGL::Shader;
	using Vcl::Graphics::Runtime::OpenGL::ShaderProgramDescription;
	using Vcl::Graphics::Runtime::OpenGL::ShaderProgram;
	using Vcl::Graphics::Runtime::BufferDescription;
	using Vcl::Graphics::Runtime::BufferInitData;
	using Vcl::Graphics::Runtime::InputLayoutDescription;
	using Vcl::Graphics::Runtime::PipelineStateDescription;
	using Vcl::Graphics::Runtime::ShaderType;
	using Vcl::Graphics::Runtime::ResourceUsage;
	using Vcl::Graphics::Runtime::VertexDataClassification;
	using Vcl::Graphics::SurfaceFormat;

	_engine = std::make_unique<Vcl::Graphics::Runtime::OpenGL::GraphicsEngine>();

	InputLayoutDescription planeLayout =
	{
		{ "PlaneEquation", SurfaceFormat::R32G32B32A32_FLOAT, 0, 0, 0, VertexDataClassification::VertexDataPerObject, 0 },
	};
	
	InputLayoutDescription opaqueTriLayout =
	{
		{ "Index",  SurfaceFormat::R32G32B32_SINT, 0, 0, 0, VertexDataClassification::VertexDataPerObject, 0 },
		{ "Colour", SurfaceFormat::R32G32B32A32_FLOAT, 0, 1, 0, VertexDataClassification::VertexDataPerObject, 0 },
	};
	
	InputLayoutDescription opaqueTetraLayout =
	{
		{ "Index",  SurfaceFormat::R32G32B32A32_SINT, 0, 0, 0, VertexDataClassification::VertexDataPerObject, 0 },
		{ "Colour", SurfaceFormat::R32G32B32A32_FLOAT, 0, 1, 0, VertexDataClassification::VertexDataPerObject, 0 },
	};

	Shader boxVert = createShader(ShaderType::VertexShader, ":/shaders/debug/boundinggrid.vert");
	Shader boxGeom = createShader(ShaderType::GeometryShader, ":/shaders/debug/boundinggrid.geom");
	Shader boxFrag = createShader(ShaderType::FragmentShader, ":/shaders/debug/staticcolour.frag");

	Shader planeVert = createShader(ShaderType::VertexShader, ":/shaders/debug/plane.vert");
	Shader planeGeom = createShader(ShaderType::GeometryShader, ":/shaders/debug/plane.geom");

	Shader opaqueTriVert = createShader(ShaderType::VertexShader, ":/shaders/trimesh.vert");
	Shader opaqueTriGeom = createShader(ShaderType::GeometryShader, ":/shaders/trimesh.geom");

	Shader opaqueTetraVert = createShader(ShaderType::VertexShader, ":/shaders/tetramesh.vert");
	Shader opaqueTetraGeom = createShader(ShaderType::GeometryShader, ":/shaders/tetramesh.geom");
	Shader opaqueTetraGeomWire = createShader(ShaderType::GeometryShader, ":/shaders/tetramesh_wireframe.geom");
	Shader opaqueTetraGeomPoints = createShader(ShaderType::GeometryShader, ":/shaders/tetramesh_sphere.geom");
	Shader idTetraVert = createShader(ShaderType::VertexShader, ":/shaders/objectid_tetramesh.vert");
	Shader idTetraGeom = createShader(ShaderType::GeometryShader, ":/shaders/objectid_tetramesh.geom");

	Shader meshFrag = createShader(ShaderType::FragmentShader, ":/shaders/mesh.frag");
	Shader idFrag = createShader(ShaderType::FragmentShader, ":/shaders/objectid.frag");

	PipelineStateDescription boxPSDesc;
	boxPSDesc.VertexShader = &boxVert;
	boxPSDesc.GeometryShader = &boxGeom;
	boxPSDesc.FragmentShader = &boxFrag;
	_boxPipelineState = Vcl::make_owner<PipelineState>(boxPSDesc);

	PipelineStateDescription planePSDesc;
	planePSDesc.InputLayout = planeLayout;
	planePSDesc.VertexShader = &planeVert;
	planePSDesc.GeometryShader = &planeGeom;
	planePSDesc.FragmentShader = &meshFrag;
	_planePipelineState = Vcl::make_owner<PipelineState>(planePSDesc);

	PipelineStateDescription idTetraPSDesc;
	idTetraPSDesc.InputLayout = opaqueTetraLayout;
	idTetraPSDesc.VertexShader = &idTetraVert;
	idTetraPSDesc.GeometryShader = &idTetraGeom;
	idTetraPSDesc.FragmentShader = &idFrag;
	_idTetraMeshPipelineState = Vcl::make_owner<PipelineState>(idTetraPSDesc);

	PipelineStateDescription opaqueTriPSDesc;
	opaqueTriPSDesc.InputLayout = opaqueTriLayout;
	opaqueTriPSDesc.VertexShader = &opaqueTriVert;
	opaqueTriPSDesc.GeometryShader = &opaqueTriGeom;
	opaqueTriPSDesc.FragmentShader = &meshFrag;
	_opaqueTriMeshPipelineState = Vcl::make_owner<PipelineState>(opaqueTriPSDesc);

	PipelineStateDescription opaqueTetraPSDesc;
	opaqueTetraPSDesc.InputLayout = opaqueTetraLayout;
	opaqueTetraPSDesc.VertexShader = &opaqueTetraVert;
	opaqueTetraPSDesc.GeometryShader = &opaqueTetraGeom;
	opaqueTetraPSDesc.FragmentShader = &meshFrag;
	_opaqueTetraMeshPipelineState = Vcl::make_owner<PipelineState>(opaqueTetraPSDesc);

	opaqueTetraPSDesc.GeometryShader = &opaqueTetraGeomWire;
	_opaqueTetraMeshWirePipelineState = Vcl::make_owner<PipelineState>(opaqueTetraPSDesc);

	opaqueTetraPSDesc.GeometryShader = &opaqueTetraGeomPoints;
	_opaqueTetraMeshPointsPipelineState = Vcl::make_owner<PipelineState>(opaqueTetraPSDesc);

	// Build up the marching cubes tables
	MarchingCubesTables mcTables;
	memcpy(mcTables.caseToNumPolys, Vcl::Geometry::caseToNumPolys, sizeof(Vcl::Geometry::caseToNumPolys));
	memcpy(mcTables.edgeVertexList, Vcl::Geometry::edgeVertexList, sizeof(Vcl::Geometry::edgeVertexList));

	BufferDescription mcDesc;
	mcDesc.Usage = ResourceUsage::Default;
	mcDesc.SizeInBytes = sizeof(MarchingCubesTables);

	BufferInitData mcData;
	mcData.Data = &mcTables;
	mcData.SizeInBytes = sizeof(MarchingCubesTables);

	_marchingCubesTables = Vcl::make_owner<Buffer>(mcDesc, false, false, &mcData);

	// Buffer for the ground plane
	BufferDescription planeDesc;
	planeDesc.Usage = ResourceUsage::Default;
	planeDesc.SizeInBytes = sizeof(Eigen::Vector4f);

	Eigen::Vector4f grouldPlane{ 0, 1, 0, -2 };
	BufferInitData planeData;
	planeData.Data = grouldPlane.data();
	planeData.SizeInBytes = sizeof(Eigen::Vector4f);

	_planeBuffer = Vcl::make_owner<Buffer>(planeDesc, false, false, &planeData);
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
		cbuffer_cam_ptr->Viewport = Eigen::Vector4f{ 0, 0, (float)_owner->width(), (float)_owner->height() };
		cbuffer_cam_ptr->Frustum = scene->frustum();
		cbuffer_cam_ptr->ViewMatrix = V;
		cbuffer_cam_ptr->ProjectionMatrix = P;

		_engine->setConstantBuffer(PER_FRAME_CAMERA_DATA_LOC, cbuffer_cam);

		// Draw the object buffer
		{
			_engine->setRenderTargets({ &_idBuffer, 1 }, _idBufferDepth);

			auto volumes = scene->entityManager()->get<GPUVolumeMesh>();
			if (!volumes->empty())
			{
				volumes->forEach([this, &M](const GPUVolumeMesh* volume_mesh)
				{
					renderTetMesh(volume_mesh, _opaqueTetraMeshPipelineState, M);
				});
			}

			// Queue a read-back
			_engine->queueReadback(_idBuffer);
		}

		// Reset the render target
		this->framebufferObject()->bind();

		// Draw the bounding grid
		{
			// Configure the layout
			_engine->setPipelineState(_boxPipelineState);

			_boxPipelineState->program().setUniform(_boxPipelineState->program().uniform("ModelMatrix"), M);

			// Render the grid
			// 3 Line-loops with 4 points, 11 replications of the loops
			glDrawArraysInstanced(GL_LINES_ADJACENCY, 0, 12, 11);
		}

		// Draw the ground
		{
			// Configure the layout
			_engine->setPipelineState(_planePipelineState);
		
			_engine->setConstantBuffer(MARCHING_CUBES_TABLES_LOC, { _marchingCubesTables, 0, _marchingCubesTables->sizeInBytes(), nullptr });
			_planePipelineState->program().setUniform(_planePipelineState->program().uniform("ModelMatrix"), M);
		
			// Bind the buffers
			glBindVertexBuffer(0, _planeBuffer->id(), 0, sizeof(Eigen::Vector4f));
		
			// Render the mesh
			glDrawArrays(GL_POINTS, 0, 1);
		}
		/*{
			std::vector<Eigen::Vector3f> points;
			Vcl::Util::computePlaneFrustumIntersection({ 0, 1, 0, -2 }, V * M, scene->frustum(), points);

			auto plane_buffer = _engine->requestPerFrameLinearMemory(points.size() * sizeof(Eigen::Vector3f));
			auto plane_buffer_ptr = reinterpret_cast<Eigen::Vector3f*>(plane_buffer.data());
			Eigen::Matrix4f T = (V * M).inverse();
			std::transform(points.begin(), points.end(), plane_buffer_ptr, [&T](const Eigen::Vector3f& v) -> Eigen::Vector3f
			{
				return (T * Eigen::Vector4f(v.x(), v.y(), v.z(), 1)).segment<3>(0);
			});

			auto plane_idx_buffer = _engine->requestPerFrameLinearMemory(points.size() * sizeof(int));
			auto plane_idx_buffer_ptr = reinterpret_cast<int*>(plane_idx_buffer.data());
			std::iota(plane_idx_buffer_ptr, plane_idx_buffer_ptr + points.size(), 0);

			auto plane_col_buffer = _engine->requestPerFrameLinearMemory(points.size() / 3 * sizeof(Eigen::Vector4f));
			auto plane_col_buffer_ptr = reinterpret_cast<Eigen::Vector4f*>(plane_col_buffer.data());
			std::fill(plane_col_buffer_ptr, plane_col_buffer_ptr + points.size() / 3, Eigen::Vector4f(0, 1, 0, 1));

			// Configure the layout
			_engine->setPipelineState(_opaqueTriMeshPipelineState);

			////////////////////////////////////////////////////////////////////
			// Render the mesh
			////////////////////////////////////////////////////////////////////

			_opaqueTriMeshPipelineState->program().setUniform(_opaqueTriMeshPipelineState->program().uniform("ModelMatrix"), M);

			// Set the vertex positions
			_opaqueTriMeshPipelineState->program().setBuffer("VertexPositions", &plane_buffer.owner(), plane_buffer.offset(), plane_buffer.size());

			// Bind the buffers
			auto& gl_plane_indices = static_cast<const Vcl::Graphics::Runtime::OpenGL::Buffer&>(plane_idx_buffer.owner());
			glBindVertexBuffer(0, gl_plane_indices.id(), plane_idx_buffer.offset(), sizeof(Eigen::Vector3i));

			auto& gl_plane_colours = static_cast<const Vcl::Graphics::Runtime::OpenGL::Buffer&>(plane_col_buffer.owner());
			glBindVertexBuffer(1, gl_plane_colours.id(), plane_col_buffer.offset(), sizeof(Eigen::Vector4f));

			// Render the mesh
			glDrawArrays(GL_POINTS, 0, (GLsizei)points.size() / 3);
		}*/

		auto surfaces = scene->entityManager()->get<GPUSurfaceMesh>();
		if (!surfaces->empty())
		{
			// Configure the layout
			_engine->setPipelineState(_opaqueTriMeshPipelineState);

			////////////////////////////////////////////////////////////////////
			// Render the mesh
			////////////////////////////////////////////////////////////////////
		
			_opaqueTriMeshPipelineState->program().setUniform(_opaqueTriMeshPipelineState->program().uniform("ModelMatrix"), M);

			surfaces->forEach([this](const GPUSurfaceMesh* surfaceMesh)
			{
				// Set the vertex positions
				_opaqueTriMeshPipelineState->program().setBuffer("VertexPositions", surfaceMesh->positions());

				// Bind the buffers
				glBindVertexBuffer(0, surfaceMesh->indices()->id(), 0, sizeof(Eigen::Vector3i));
				glBindVertexBuffer(1, surfaceMesh->faceColours()->id(), 0, sizeof(Eigen::Vector4f));

				// Render the mesh
				glDrawArrays(GL_POINTS, 0, (GLsizei)surfaceMesh->nrFaces());
			});
		}

		auto volumes = scene->entityManager()->get<GPUVolumeMesh>();
		if (!volumes->empty())
		{
			volumes->forEach([this, &M](const GPUVolumeMesh* volume_mesh)
			{
				if (_renderWireframe)
				{
					renderTetMesh(volume_mesh, _opaqueTetraMeshPointsPipelineState, M);
					renderTetMesh(volume_mesh, _opaqueTetraMeshWirePipelineState, M);
				}
				else
				{
					renderTetMesh(volume_mesh, _opaqueTetraMeshPipelineState, M);
				}
			});
		}
	}

	_engine->endFrame();
	_owner->window()->resetOpenGLState();
	update();
}

void FboRenderer::renderTetMesh(const GPUVolumeMesh* mesh, Vcl::ref_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> ps, const Eigen::Matrix4f& M)
{
	// Configure the state
	_engine->setPipelineState(ps);

	////////////////////////////////////////////////////////////////////
	// Render the mesh
	////////////////////////////////////////////////////////////////////
	ps->program().setUniform(ps->program().uniform("ModelMatrix"), M);

	// Set the vertex positions
	ps->program().setBuffer("VertexPositions", mesh->positions());

	// Bind the buffers
	glBindVertexBuffer(0, mesh->indices()->id(), 0, sizeof(Eigen::Vector4i));
	glBindVertexBuffer(1, mesh->volumeColours()->id(), 0, sizeof(Eigen::Vector4f));

	// Render the mesh
	glDrawArrays(GL_POINTS, 0, (GLsizei)mesh->nrVolumes());
}

void FboRenderer::synchronize(QQuickFramebufferObject* item)
{
	auto* view = dynamic_cast<MeshView*>(item);
	if (view)
	{
		_owner = view;
		_owner->scene()->setEngine(_engine.get());

		_renderWireframe = _owner->renderWireframe();
	}

	if (_owner && _owner->scene())
	{
		_owner->scene()->update();
	}
}

QOpenGLFramebufferObject* FboRenderer::createFramebufferObject(const QSize &size)
{
	// Clean-up the old render-targets
	if (_idBuffer)
	{
		_engine->deletePersistentTexture(_idBuffer);
	}

	// Create the new render-targets
	Vcl::Graphics::Runtime::Texture2DDescription desc;
	desc.Width = size.width();
	desc.Height = size.height();
	desc.Format = Vcl::Graphics::SurfaceFormat::R32G32_SINT;
	desc.ArraySize = 1;
	desc.MipLevels = 1;
	_idBuffer = _engine->allocatePersistentTexture(std::make_unique<Vcl::Graphics::Runtime::OpenGL::Texture2D>(desc));

	desc.Format = Vcl::Graphics::SurfaceFormat::D32_FLOAT;
	_idBufferDepth = Vcl::make_owner<Vcl::Graphics::Runtime::OpenGL::Texture2D>(desc);

	QOpenGLFramebufferObjectFormat format;
	format.setAttachment(QOpenGLFramebufferObject::Depth);
	format.setSamples(4);
	return new QOpenGLFramebufferObject(size, format);
}

MeshView::MeshView(QQuickItem* parent)
: QQuickFramebufferObject(parent)
{
	setMirrorVertically(true);
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
