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

// Qt
#include <QtCore/QObject>

// VCL
#include <vcl/components/entitymanager.h>
#include <vcl/graphics/runtime/graphicsengine.h>
#include <vcl/graphics/camera.h>
#include <vcl/graphics/trackballcameracontroller.h>

#include "components/gpusurfacemesh.h"
#include "components/gpuvolumemesh.h"
#include "components/meshstatisticscomponent.h"
#include "components/transform.h"

#include "editor/componentadapter.h"
#include "editor/entityadapter.h"

/*!
 *	\note Combination of model and view-model
 */
class Scene : public QObject
{
	Q_OBJECT

	Q_PROPERTY(Editor::EntityAdapterModel* entityModel READ entityModel NOTIFY entityModelChanged)

public:
	Scene(QObject* parent = 0);
	~Scene();

public:
	void setEngine(Vcl::Graphics::Runtime::GraphicsEngine* engine) { _engine = engine; }

public:
	void update();

public:
	const Vcl::Components::EntityManager* entityManager() const { return &_entityManager; }

public:
	Vcl::Graphics::Camera* camera() const { return _camera; }

public slots :
	void createSurfaceSphere();
	void createBar(int x, int y, int z);
	void loadMesh(const QUrl& path);

public slots:
	void startRotate(float ratio_x, float ratio_y);
	void rotate(float ratio_x, float ratio_y);
	void endRotate();

public:
	const Eigen::AlignedBox3f& boundingBox() const { return _boundingBox; }

	const Eigen::Vector4f& frustum() const { return _frustumData; }
	const Eigen::Matrix4f& modelMatrix() const { return _modelMatrix; }
	const Eigen::Matrix4f& viewMatrix() const { return _viewMatrix; }
	const Eigen::Matrix4f& projMatrix() const { return _projMatrix; }

public: // Editor support
	Editor::EntityAdapterModel* entityModel();

signals:
	void entityModelChanged();

private:
	void initializeTetraMesh(std::unique_ptr<Vcl::Geometry::TetraMesh> mesh);
	void updateBoundingBox();

private: // Engine
	Vcl::Graphics::Runtime::GraphicsEngine* _engine{ nullptr };

private: // Scene data
	Vcl::Graphics::TrackballCameraController _cameraController;

	/// Bounding box of the scene
	Eigen::AlignedBox3f _sceneBoundingBox;
	
private: // Render data

	Eigen::AlignedBox3f _boundingBox;

	Eigen::Vector4f _frustumData;

	Eigen::Matrix4f _modelMatrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f _viewMatrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f _projMatrix = Eigen::Matrix4f::Identity();

private: // Entities
	//! Entity manager
	Vcl::Components::EntityManager _entityManager;

	//! QML exposure of scene entities
	Editor::EntityAdapterModel _entityAdapterModel;

private: // Camera entity
	Vcl::Components::Entity _cameraEntity;
	Vcl::Graphics::Camera* _camera;

private: // Mesh entities
	std::vector<Vcl::Components::Entity> _meshes;
};
