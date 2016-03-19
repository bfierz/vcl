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
	GPUSurfaceMesh(std::unique_ptr<Vcl::Geometry::TriMesh> mesh);

public:
	size_t nrFaces() const { return _triMesh->nrFaces(); }

	Vcl::Graphics::Runtime::OpenGL::Buffer* indices()       const { return _indices.get(); }
	Vcl::Graphics::Runtime::OpenGL::Buffer* positions()     const { return _positions.get(); }
	Vcl::Graphics::Runtime::OpenGL::Buffer* faceColours() const { return _volumeColours.get(); }

private:
	std::unique_ptr<Vcl::Geometry::TriMesh> _triMesh;

	//! Index structure
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _indices;

	//! Position data
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _positions;

	//! Volume-colour data
	std::unique_ptr<Vcl::Graphics::Runtime::OpenGL::Buffer> _volumeColours;
};
