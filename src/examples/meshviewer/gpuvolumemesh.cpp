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
#include <vcl/graphics/runtime/opengl/resource/buffer.h>

GPUVolumeMesh::GPUVolumeMesh(std::unique_ptr<Vcl::Geometry::TetraMesh> mesh)
: _tetraMesh(std::move(mesh))
{
	using namespace Vcl::Geometry;
	using namespace Vcl::Graphics::Runtime;

	// Create the index buffer
	BufferDescription idxDesc;
	idxDesc.Usage = Usage::Default;
	idxDesc.SizeInBytes = _tetraMesh->nrVolumes() * sizeof(IndexDescriptionTrait<TetraMesh>::Volume);

	BufferInitData idxData;
	idxData.Data = _tetraMesh->volumes()->data();
	idxData.SizeInBytes = _tetraMesh->nrVolumes() * sizeof(IndexDescriptionTrait<TetraMesh>::Volume);

	_indices = std::make_unique<OpenGL::Buffer>(idxDesc, false, false, &idxData);

	// Create the position buffer
	BufferDescription posDesc;
	posDesc.CPUAccess = CPUAccess::Write;
	posDesc.Usage = Usage::Default;
	posDesc.SizeInBytes = _tetraMesh->nrVertices() * sizeof(IndexDescriptionTrait<TetraMesh>::Vertex);

	BufferInitData posData;
	posData.Data = _tetraMesh->vertices()->data();
	posData.SizeInBytes = _tetraMesh->nrVertices() * sizeof(IndexDescriptionTrait<TetraMesh>::Vertex);

	_positions = std::make_unique<OpenGL::Buffer>(posDesc, false, false, &posData);

	// Create the volume-colour buffer
	auto colours = _tetraMesh->addVolumeProperty<Eigen::Vector4f>("Colour", Eigen::Vector4f{ 0.2f, 0.8f, 0.2f, 1 });

	BufferDescription colDesc;
	colDesc.CPUAccess = CPUAccess::Write;
	colDesc.Usage = Usage::Default;
	colDesc.SizeInBytes = _tetraMesh->nrVolumes() * sizeof(Eigen::Vector4f);

	BufferInitData colData;
	colData.Data = colours->data();
	colData.SizeInBytes = _tetraMesh->nrVolumes() * sizeof(Eigen::Vector4f);

	_volumeColours = std::make_unique<OpenGL::Buffer>(colDesc, false, false, &colData);
}

GPUVolumeMesh::~GPUVolumeMesh()
{
}
