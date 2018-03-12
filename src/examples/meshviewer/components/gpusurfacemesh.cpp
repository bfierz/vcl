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
#include "gpusurfacemesh.h"

// VCL
#include <vcl/geometry/meshoperations.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>

GPUSurfaceMesh::GPUSurfaceMesh(Vcl::Geometry::TriMesh* mesh)
: _triMesh(mesh)
{
	update();
}

void GPUSurfaceMesh::update()
{
	using namespace Vcl::Geometry;
	using namespace Vcl::Graphics::Runtime;

	_nrSurfaceElements = _triMesh->nrFaces();

	// Convert the triangle list to a triangle-adjacency list
	const auto tri_adjs = convertToTriangleAdjacency<IndexDescriptionTrait<TriMesh>::VertexId>({ _triMesh->faces()->data(), _triMesh->nrFaces() });
	_indexStride = sizeof(decltype(tri_adjs)::value_type);

	// Create the index buffer
	BufferDescription idxDesc;
	idxDesc.Usage = ResourceUsage::Default;
	idxDesc.SizeInBytes = static_cast<uint32_t>(tri_adjs.size() * _indexStride);

	BufferInitData idxData;
	idxData.Data = tri_adjs.data();
	idxData.SizeInBytes = static_cast<uint32_t>(tri_adjs.size() * _indexStride);

	_indices = std::make_unique<OpenGL::Buffer>(idxDesc, false, false, &idxData);

	// Create the position buffer
	_positionStride = sizeof(IndexDescriptionTrait<TriMesh>::Vertex);

	BufferDescription posDesc;
	posDesc.CPUAccess = ResourceAccess::Write;
	posDesc.Usage = ResourceUsage::Default;
	posDesc.SizeInBytes = static_cast<uint32_t>(_triMesh->nrVertices() * _positionStride);

	BufferInitData posData;
	posData.Data = _triMesh->vertices()->data();
	posData.SizeInBytes = static_cast<uint32_t>(_triMesh->nrVertices() * _positionStride);

	_positions = std::make_unique<OpenGL::Buffer>(posDesc, false, false, &posData);

	// Create the normal buffer
	auto normals = _triMesh->vertexProperty<Eigen::Vector3f>("Normals");
	_normalStride = sizeof(Eigen::Vector3f);

	BufferDescription normalDesc;
	normalDesc.CPUAccess = ResourceAccess::Write;
	normalDesc.Usage = ResourceUsage::Default;
	normalDesc.SizeInBytes = static_cast<uint32_t>(normals->size() * _normalStride);

	BufferInitData normalData;
	normalData.Data = normals->data();
	normalData.SizeInBytes = static_cast<uint32_t>(normals->size() * _normalStride);

	_normals = std::make_unique<OpenGL::Buffer>(normalDesc, false, false, &normalData);

	// Create the volume-colour buffer
	auto colours = _triMesh->addFaceProperty<Eigen::Vector4f>("Colour", Eigen::Vector4f{ 0.2f, 0.8f, 0.2f, 1 });

	BufferDescription colDesc;
	colDesc.CPUAccess = ResourceAccess::Write;
	colDesc.Usage = ResourceUsage::Default;
	colDesc.SizeInBytes = _triMesh->nrFaces() * sizeof(Eigen::Vector4f);

	BufferInitData colData;
	colData.Data = colours->data();
	colData.SizeInBytes = _triMesh->nrFaces() * sizeof(Eigen::Vector4f);

	_volumeColours = std::make_unique<OpenGL::Buffer>(colDesc, false, false, &colData);
}
