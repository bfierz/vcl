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
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::GraphicsEngine> _engine;

private:
	MeshView* _owner{ nullptr };

private: // States
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _planePipelineState;

	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _opaqueTriMeshPipelineState;
	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _opaqueTetraMeshPipelineState;

	Vcl::owner_ptr<Vcl::Graphics::Runtime::OpenGL::PipelineState> _idTetraMeshPipelineState;

private: // Render targets
	Vcl::ref_ptr<Vcl::Graphics::Runtime::DynamicTexture<3>> _idBuffer;
	Vcl::owner_ptr<Vcl::Graphics::Runtime::Texture> _idBufferDepth;

private: // Static geometry
	Vcl::ref_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _planeBuffer;
};

class MeshView : public QQuickFramebufferObject
{
	Q_OBJECT

	// Properties
	Q_PROPERTY(Scene* scene READ scene WRITE setScene)

public:
	MeshView(QQuickItem *parent = Q_NULLPTR);

public:
	Scene* scene() const { return _scene; }
	void setScene(Scene* s)
	{
		_scene = s;
		update();
	}

private: // Implementations
	Renderer* createRenderer() const override;
	void geometryChanged(const QRectF & newGeometry, const QRectF & oldGeometry) override;

private:
	Scene* _scene{ nullptr };
};
