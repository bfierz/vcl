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

// VCL
#include <vcl/geometry/tetramesh.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>

class GPUVolumeMesh
{
public:
	GPUVolumeMesh(Vcl::Geometry::TetraMesh* mesh);
	~GPUVolumeMesh();

public:
	size_t nrSurfaceFaces() const { return _tetraMesh->nrSurfaceFaces(); }
	size_t nrVolumes() const { return _tetraMesh->nrVolumes(); }

	Vcl::Graphics::Runtime::OpenGL::Buffer* indices() const { return _indices.get(); }
	Vcl::Graphics::Runtime::OpenGL::Buffer* positions() const { return _positions.get(); }
	Vcl::Graphics::Runtime::OpenGL::Buffer* volumeColours() const { return _volumeColours.get(); }

	Vcl::Graphics::Runtime::OpenGL::Buffer* surfaceIndices() const { return _surfaceIndices.get(); }
	Vcl::Graphics::Runtime::OpenGL::Buffer* surfaceNormals() const { return _surfaceNormals.get(); }
	Vcl::Graphics::Runtime::OpenGL::Buffer* surfaceColours() const { return _surfaceColours.get(); }

	size_t indexStride() const { return _indexStride; }
	size_t surfaceIndexStride() const { return _surfaceIndexStride; }
	size_t positionStride() const { return _positionStride; }

private:
	Vcl::Geometry::TetraMesh* _tetraMesh;

	//! Index structure
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _indices;

	//! Surface index structure
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _surfaceIndices;

	//! Position data
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _positions;

	//! Volume-colour data
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _volumeColours;

	//! Surface normal data
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _surfaceNormals;

	//! Surface colour data
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _surfaceColours;
	//! Stride between two tetrahedra
	size_t _indexStride{ 0 };

	//! Stride between two surface elements
	size_t _surfaceIndexStride{ 0 };

	//! Stride between two positions
	size_t _positionStride{ 0 };
};
