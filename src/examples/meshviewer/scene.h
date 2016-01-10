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
#include <vcl/graphics/camera.h>
#include <vcl/graphics/trackballcameracontroller.h>

#include "gpusurfacemesh.h"
#include "gpuvolumemesh.h"

/*!
 *	\note Combination of model and view-model
 */
class Scene : public QObject
{
	Q_OBJECT

public:
	Scene(QObject* parent = 0);
	~Scene();

public:
	void update();

public:
	Vcl::Graphics::Camera* camera() const { return _camera.get(); }

public slots :
	void createSurfaceSphere();
	void createBar(int x, int y, int z);
	void loadMesh(const QUrl& path);

public slots:
	void startRotate(float ratio_x, float ratio_y);
	void rotate(float ratio_x, float ratio_y);
	void endRotate();

public:
	const Eigen::Matrix4f& modelMatrix() const { return _modelMatrix; }
	const Eigen::Matrix4f& viewMatrix() const { return _viewMatrix; }
	const Eigen::Matrix4f& projMatrix() const { return _projMatrix; }

	GPUSurfaceMesh* surfaceMesh() const { return _surfaceMesh.get(); }
	GPUVolumeMesh* volumeMesh() const { return _volumeMesh.get(); }

private: // Update data
	std::unique_ptr<Vcl::Geometry::TriMesh> _triMesh;
	std::unique_ptr<Vcl::Geometry::TetraMesh> _tetraMesh;

	std::unique_ptr<Vcl::Graphics::Camera> _camera;
	Vcl::Graphics::TrackballCameraController _cameraController;
	
private: // Render data

	Eigen::Matrix4f _modelMatrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f _viewMatrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f _projMatrix = Eigen::Matrix4f::Identity();

	std::unique_ptr<GPUSurfaceMesh> _surfaceMesh;
	std::unique_ptr<GPUVolumeMesh> _volumeMesh;
};
