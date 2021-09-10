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
#include <vcl/geometry/trimesh.h>
#include <vcl/graphics/runtime/opengl/resource/buffer.h>

class GPUSurfaceMesh
{
public:
	GPUSurfaceMesh(Vcl::Geometry::TriMesh* mesh);

	void update();

public:
	size_t nrFaces() const { return _nrSurfaceElements; }

	Vcl::Graphics::Runtime::OpenGL::Buffer* indices() const { return _indices.get(); }
	Vcl::Graphics::Runtime::OpenGL::Buffer* positions() const { return _positions.get(); }
	Vcl::Graphics::Runtime::OpenGL::Buffer* normals() const { return _normals.get(); }
	Vcl::Graphics::Runtime::OpenGL::Buffer* faceColours() const { return _volumeColours.get(); }

	size_t indexStride() const { return _indexStride; }
	size_t positionStride() const { return _positionStride; }
	size_t normalStride() const { return _normalStride; }

private:
	Vcl::Geometry::TriMesh* _triMesh;

	//! Index structure
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _indices;

	//! Position data
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _positions;

	//! Normal data
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _normals;

	//! Volume-colour data
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _volumeColours;

	//! Number of faces of the surface
	size_t _nrSurfaceElements{ 0 };

	//! Stride between two primitives
	size_t _indexStride{ 0 };

	//! Stride between two positions
	size_t _positionStride{ 0 };

	//! Stride between two normals
	size_t _normalStride{ 0 };
};
