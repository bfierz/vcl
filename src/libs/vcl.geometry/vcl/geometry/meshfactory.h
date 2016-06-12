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

// VCL
#include <vcl/geometry/tetramesh.h>
#include <vcl/geometry/trimesh.h>

namespace Vcl { namespace Geometry
{
	template<typename Mesh>
	class MeshFactory
	{
	public:
		static std::unique_ptr<Mesh> createHomogenousCubes(unsigned int count_x = 1, unsigned int count_y = 1, unsigned int count_z = 1);

		static std::unique_ptr<Mesh> loadMesh(const std::string& path);
	};

	template<>
	class MeshFactory<TetraMesh>
	{
	public:
		static std::unique_ptr<TetraMesh> createHomogenousCubes(unsigned int count_x = 1, unsigned int count_y = 1, unsigned int count_z = 1);

		static std::unique_ptr<TetraMesh> loadMesh(const std::string& path);
	};

	class TriMeshFactory
	{
	public:
		static std::unique_ptr<TriMesh> createSphere(const Vector3f& center, float radius, unsigned int stacks, unsigned int slices, bool inverted);

	};
}}
