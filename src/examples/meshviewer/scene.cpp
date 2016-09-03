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

// Qt
#include <QtCore/QUrl>

// VCL
#include <vcl/geometry/io/serialiser_nvidia_tet_file.h>
#include <vcl/geometry/io/serialiser_tetgen.h>
#include <vcl/geometry/io/tetramesh_serialiser.h>
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/tetramesh.h>

Scene::Scene(QObject* parent)
: QObject(parent)
{
	using namespace Vcl::Graphics;

	// Register the camera as a component
	_entityManager.registerComponent<Camera>();
	_entityManager.registerComponent<Vcl::Geometry::TetraMesh>();
	_entityManager.registerComponent<Vcl::Geometry::TriMesh>();
	_entityManager.registerComponent<GPUSurfaceMesh>();
	_entityManager.registerComponent<GPUVolumeMesh>();

	// Create a new camera
	_cameraEntity = _entityManager.create();
	_camera = _entityManager.create<Camera>(_cameraEntity, std::make_shared<OpenGL::MatrixFactory>());
	_cameraController.setCamera(_camera);
}
Scene::~Scene()
{
	_entityManager.destroy(_cameraEntity);
}

void Scene::update()
{
	_frustumData = { tan(_camera->fieldOfView() / 2.0f), (float) _camera->viewportWidth() / (float)_camera->viewportHeight(), _camera->nearPlane(), _camera->farPlane() };

	_modelMatrix = _cameraController.currObjectTransformation();
	_viewMatrix = _camera->view();
	_projMatrix = _camera->projection();
}

void Scene::createSurfaceSphere()
{
	using namespace Vcl::Geometry;

	std::cout << "Creating sphere mesh" << std::endl;

	// Create a new entity
	auto mesh_entity = _entityManager.create();
	_meshes.push_back(mesh_entity);

	auto mesh = TriMeshFactory::createSphere({ 0, 0, 0 }, 1, 10, 10, false);

	// Compute the new camera configuration
	Eigen::AlignedBox3f bb;
	auto vertices = mesh->vertices();
	for (size_t i = 0, end = vertices->size(); i < end; i++)
	{
		bb.extend(vertices[i]);
	}
	_camera->encloseInFrustum(bb.center(), { 0, 0, 1 }, bb.diagonal().norm());
	_cameraController.setRotationCenter(bb.center());

	// Create the mesh component
	auto mesh_component = _entityManager.create<TriMesh>(mesh_entity, std::move(*mesh));

	// Create GPU buffers
	_engine->enqueueCommand([this, mesh_entity, mesh_component]()
	{
		_entityManager.create<GPUSurfaceMesh>(mesh_entity, mesh_component);
	});
}

void Scene::createBar(int x, int y, int z)
{
	using namespace Vcl::Geometry;

	std::cout << "Creating bar mesh of resolution (" << x << ", " << y << ", " << z << ")" << std::endl;

	auto mesh = MeshFactory<TetraMesh>::createHomogenousCubes(x, y, z);

	initializeTetraMesh(std::move(mesh));
}

void Scene::loadMesh(const QUrl& path)
{
	using namespace Vcl::Geometry::IO;

	std::cout << "Load mesh: " << path.toLocalFile().toUtf8().data() << std::endl;

	TetraMeshDeserialiser deserialiser;

	if (path.toString().endsWith(".tet"))
	{
		NvidiaTetSerialiser loader;
		loader.load(&deserialiser, path.toLocalFile().toUtf8().data());
	}
	else if (path.toString().endsWith(".ele") || path.toString().endsWith(".node"))
	{
		TetGenSerialiser loader;
		loader.load(&deserialiser, path.toLocalFile().toUtf8().data());
	}
	else
	{
		return;
	}

	auto mesh = deserialiser.fetch();
	initializeTetraMesh(std::move(mesh));
}

void Scene::initializeTetraMesh(std::unique_ptr<Vcl::Geometry::TetraMesh> mesh)
{
	using namespace Vcl::Geometry;

	// Create a new entity
	auto mesh_entity = _entityManager.create();
	_meshes.push_back(mesh_entity);

	// Compute the new camera configuration
	Eigen::AlignedBox3f bb;
	auto vertices = mesh->vertices();
	for (size_t i = 0, end = vertices->size(); i < end; i++)
	{
		bb.extend(vertices[i]);
	}
	_camera->encloseInFrustum(bb.center(), { 0, 0, 1 }, bb.diagonal().norm());
	_cameraController.setRotationCenter(bb.center());

	// Create the mesh component
	auto mesh_component = _entityManager.create<TetraMesh>(mesh_entity, std::move(*mesh));

	// Create GPU buffers
	_engine->enqueueCommand([this, mesh_entity, mesh_component]()
	{
		_entityManager.create<GPUVolumeMesh>(mesh_entity, mesh_component);
	});
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

