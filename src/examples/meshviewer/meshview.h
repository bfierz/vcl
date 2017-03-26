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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/opengl.h>

// C++ standard library
#include <iostream>

// Qt
#include <QtGui/QOpenGLFramebufferObject>
#include <QtQuick/QQuickFramebufferObject>
#include <QtQuick/QSGSimpleTextureNode>

// VCL
#include <vcl/graphics/runtime/opengl/state/inputlayout.h>
#include <vcl/graphics/runtime/opengl/state/pipelinestate.h>
#include <vcl/graphics/runtime/opengl/state/shaderprogram.h>
#include <vcl/graphics/runtime/opengl/graphicsengine.h>
#include <vcl/graphics/runtime/dynamictexture.h>
#include <vcl/graphics/runtime/framebuffer.h>

#include "util/positionmanipulator.h"
#include "scene.h"

class MeshView;

class FboRenderer : public QQuickFramebufferObject::Renderer
{
public:
	FboRenderer();

public:
	void render() override;
	void synchronize(QQuickFramebufferObject* item) override;

	QOpenGLFramebufferObject* createFramebufferObject(const QSize &size);

private:
	void renderHandle(const Eigen::Matrix4f& M);
	void renderBoundingBox(const Eigen::AlignedBox3f& bb, unsigned int resolution, Vcl::ref_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> ps, const Eigen::Matrix4f& M);
	void renderTriMesh(const GPUSurfaceMesh* mesh, Vcl::ref_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> ps, const Eigen::Matrix4f& M);
	void renderTriMesh(const GPUVolumeMesh* mesh, Vcl::ref_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> ps, const Eigen::Matrix4f& M);
	void renderTetMesh(const GPUVolumeMesh* mesh, Vcl::ref_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> ps, const Eigen::Matrix4f& M);

private:
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::GraphicsEngine> _engine;

private:
	MeshView* _owner{ nullptr };

	bool _renderWireframe{ false };

private: // States
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _boxPipelineState;
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _planePipelineState;

	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _opaqueTriMeshPipelineState;
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _oulineTriMeshPS;

	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _opaqueTetraMeshPipelineState;
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _opaqueTetraMeshWirePipelineState;
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _opaqueTetraMeshPointsPipelineState;

	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _idTetraMeshPipelineState;
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _idTriMeshPipelineState;

private: // Render targets

	//! Store id of the rendered geometry
	Vcl::owner_ptr<Vcl::Graphics::Runtime::GBuffer> _idBuffer;

	//! Store the on the host side
	std::unique_ptr<Eigen::Vector2i[]> _idBufferHost;

	//! Stores all the rendered fragments for OIT
	Vcl::owner_ptr<Vcl::Graphics::Runtime::ABuffer> _transparencyBuffer;

private: // Support buffers
	Vcl::ref_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _marchingCubesTables;

private: // Static geometry
	Vcl::ref_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _planeBuffer;

private: // Helpers
	std::unique_ptr<Vcl::Editor::Util::PositionManipulator> _posManip;
};

class MeshView : public QQuickFramebufferObject
{
	Q_OBJECT

	// Properties
	Q_PROPERTY(Scene* scene READ scene WRITE setScene)
	Q_PROPERTY(bool renderWireframe READ renderWireframe WRITE setRenderWireframe)

public:
	MeshView(QQuickItem *parent = Q_NULLPTR);

public:
	Q_INVOKABLE QPoint selectObject(int x, int y);

	Q_INVOKABLE void beginDrag(int x, int y);
	Q_INVOKABLE void dragObject(int x, int y);
	Q_INVOKABLE void endDrag();

public:
	Scene* scene() const { return _scene; }
	void setScene(Scene* s)
	{
		_scene = s;
		update();
	}

	bool renderWireframe() const { return _renderWireframe; }
	void setRenderWireframe(bool wireframe)
	{
		_renderWireframe = wireframe;
		update();
	}

	const Eigen::Vector2i* idBuffer() const { return _idBuffer.get(); }
	void syncIdBuffer(std::unique_ptr<Eigen::Vector2i[]>& data, uint32_t width, uint32_t height);

private: // Implementations
	Renderer* createRenderer() const override;
	void geometryChanged(const QRectF & newGeometry, const QRectF & oldGeometry) override;

private:
	Scene* _scene{ nullptr };
	bool _renderWireframe{ false };

private: // ID buffer

	//! Store the on the host side
	std::unique_ptr<Eigen::Vector2i[]> _idBuffer;

	//! ID buffer width
	uint32_t _idBufferWidth{ 0 };

	//! ID buffer height
	uint32_t _idBufferHeight{ 0 };

private: // Position manipulator
	
	bool _manip_translation{ false };

	Eigen::Vector3f _curr_view_dir;

};
