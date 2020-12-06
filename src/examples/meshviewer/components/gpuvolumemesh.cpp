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
#include "gpuvolumemesh.h"

// VCL
#include <vcl/geometry/meshoperations.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>

namespace
{
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> createBuffer(const void* buffer, size_t nr_elements, size_t stride)
	{
		using namespace Vcl::Graphics::Runtime;

		BufferDescription desc;
		desc.Usage = BufferUsage::Vertex | BufferUsage::Index;
		desc.SizeInBytes = static_cast<uint32_t>(nr_elements * stride);

		BufferInitData data;
		data.Data = buffer;
		data.SizeInBytes = static_cast<uint32_t>(nr_elements * stride);

		return std::make_unique<Vcl::Graphics::Runtime::OpenGL::Buffer>(desc, &data);
	}
}

GPUVolumeMesh::GPUVolumeMesh(Vcl::Geometry::TetraMesh* mesh)
: _tetraMesh(mesh)
{
	using namespace Vcl::Geometry;
	using namespace Vcl::Graphics::Runtime;

	// Extract the surface of the tetra mesh
	_tetraMesh->recomputeSurface();

	// Create the index buffer
	_indexStride = sizeof(IndexDescriptionTrait<TetraMesh>::Volume);
	_indices = createBuffer(_tetraMesh->volumes()->data(), _tetraMesh->nrVolumes(), _indexStride);

	// Create the index buffer for the surface
	const auto tri_adjs = convertToTriangleAdjacency<IndexDescriptionTrait<TetraMesh>::VertexId>({ _tetraMesh->surfaceFaces()->data(), _tetraMesh->nrSurfaceFaces() });
	_surfaceIndexStride = sizeof(decltype(tri_adjs)::value_type);
	_surfaceIndices = createBuffer(tri_adjs.data(), tri_adjs.size(), _surfaceIndexStride);
	
	// Create the position buffer
	_positionStride = sizeof(IndexDescriptionTrait<TetraMesh>::Vertex);
	_positions = createBuffer(_tetraMesh->vertices()->data(), _tetraMesh->nrVertices(), _positionStride);

	// Create the volume-colour buffer
	auto colours = _tetraMesh->addVolumeProperty<Eigen::Vector4f>("Colour", Eigen::Vector4f{ 0.2f, 0.8f, 0.2f, 1 });
	_volumeColours = createBuffer(colours->data(), _tetraMesh->nrVolumes(), sizeof(Eigen::Vector4f));

	// Create the surface normal buffer
	auto face_normals = _tetraMesh->addVertexProperty<Eigen::Vector3f>("SurfaceNormals", Eigen::Vector3f{ 0.0f, 0.0f, 0.0f });
	computeNormals<IndexDescriptionTrait<TetraMesh>::VertexId>(
		{ _tetraMesh->surfaceFaces()->data(), _tetraMesh->nrSurfaceFaces() },
		{ _tetraMesh->vertices()->data(), _tetraMesh->nrVertices() },
		{ face_normals->data(), _tetraMesh->nrVertices() }
	);
	_surfaceNormals = createBuffer(face_normals->data(), face_normals->size(), sizeof(Eigen::Vector3f));

	// Create the face-colour buffer
	auto face_colours = _tetraMesh->addSurfaceProperty<Eigen::Vector4f>("Colour", Eigen::Vector4f{ 0.2f, 0.8f, 0.2f, 1 });
	_surfaceColours = createBuffer(face_colours->data(), _tetraMesh->nrSurfaceFaces(), sizeof(Eigen::Vector4f));
}

GPUVolumeMesh::~GPUVolumeMesh()
{
}
