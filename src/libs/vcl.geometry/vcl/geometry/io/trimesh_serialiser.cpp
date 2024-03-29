/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2021 Basil Fierz
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
#include <vcl/geometry/io/trimesh_serialiser.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Geometry { namespace IO {
	void TriMeshDeserialiser::begin()
	{
	}

	void TriMeshDeserialiser::end()
	{
		_mesh = std::make_unique<TriMesh>(_positions, _faces);
	}

	void TriMeshDeserialiser::sizeHintNodes(unsigned int hint)
	{
		_positions.reserve(hint);
	}

	void TriMeshDeserialiser::sizeHintEdges(unsigned int)
	{
	}

	void TriMeshDeserialiser::sizeHintFaces(unsigned int hint)
	{
		_faces.reserve(hint);
	}

	void TriMeshDeserialiser::sizeHintVolumes(unsigned int)
	{
	}

	void TriMeshDeserialiser::addNode(const std::vector<float>& coordinates)
	{
		VclRequire(coordinates.size() == 3, "Position in 3D space.");

		_positions.emplace_back(coordinates[0], coordinates[1], coordinates[2]);
	}

	void TriMeshDeserialiser::addEdge(const std::vector<unsigned int>&)
	{
	}

	void TriMeshDeserialiser::addFace(const std::vector<unsigned int>& indices)
	{
		VclRequire(indices.size() == 3, "Indices describe a triang.");

		std::array<unsigned int, 3> face = { indices[0], indices[1], indices[2] };
		_faces.push_back(face);
	}

	void TriMeshDeserialiser::addVolume(const std::vector<unsigned int>&)
	{
	}

	void TriMeshDeserialiser::addNormal(const Vector3f&)
	{
	}
}}}
