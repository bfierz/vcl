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
#include "scene.h"

// C++ standard library
#include <iostream>

// VCL
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/tetramesh.h>

Scene::Scene(QObject* parent)
: QObject(parent)
{
	using namespace Vcl::Graphics;

	_camera = std::make_unique<Camera>(std::make_shared<OpenGL::MatrixFactory>());
	_cameraController.setCamera(_camera.get());
	//_cameraController.setMode(CameraMode::CameraTarget);
}
Scene::~Scene()
{

}

void Scene::update()
{
	if (_tetraMesh)
	{
		_volumeMesh = std::make_unique<GPUVolumeMesh>(std::move(_tetraMesh));
	}

	_modelMatrix = _cameraController.currObjectTransformation();
	_viewMatrix = _camera->view();
	_projMatrix = _camera->projection();
}


void Scene::createBar(int x, int y, int z)
{
	using namespace Vcl::Geometry;

	std::cout << "Creating bar mesh of resolution (" << x << ", " << y << ", " << z << ")" << std::endl;

	_tetraMesh = MeshFactory<TetraMesh>::createHomogenousCubes(x, y, z);


	// Compute the new camera configuration
	Eigen::AlignedBox3f bb;
	auto vertices = _tetraMesh->vertices();
	for (size_t i = 0, end = vertices->size(); i < end; i++)
	{
		bb.extend(vertices[i]);
	}
	_camera->encloseInFrustum(bb.center(), { 0, 0, -1 }, bb.diagonal().norm());
}

void Scene::loadMesh(const QUrl& path)
{

}

void Scene::startRotate(float ratio_x, float ratio_y)
{
	_cameraController.startRotate(ratio_x, ratio_y);
}
void Scene::rotate(float ratio_x, float ratio_y)
{
	_cameraController.rotate(ratio_x, ratio_y);
}
void Scene::endRotate()
{
	_cameraController.endRotate();
}

