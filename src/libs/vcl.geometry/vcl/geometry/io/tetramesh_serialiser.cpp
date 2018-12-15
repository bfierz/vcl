/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#include <vcl/geometry/io/tetramesh_serialiser.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Geometry { namespace IO
{
	void TetraMeshDeserialiser::begin()
	{
	}

	void TetraMeshDeserialiser::end()
	{
		_mesh = std::make_unique<TetraMesh>(_positions, _volumes);
	}

	void TetraMeshDeserialiser::sizeHintNodes(unsigned int hint)
	{
		_positions.reserve(hint);
	}

	void TetraMeshDeserialiser::sizeHintEdges(unsigned int)
	{
	}

	void TetraMeshDeserialiser::sizeHintFaces(unsigned int)
	{
	}

	void TetraMeshDeserialiser::sizeHintVolumes(unsigned int hint)
	{
		_volumes.reserve(hint);
	}

	void TetraMeshDeserialiser::addNode(const std::vector<float>& coordinates)
	{
		VclRequire(coordinates.size() == 3, "Position in 3D space.");

		_positions.emplace_back(coordinates[0], coordinates[1], coordinates[2]);
	}

	void TetraMeshDeserialiser::addEdge(const std::vector<unsigned int>&)
	{
	}

	void TetraMeshDeserialiser::addFace(const std::vector<unsigned int>&)
	{
	}

	void TetraMeshDeserialiser::addVolume(const std::vector<unsigned int>& indices)
	{
		VclRequire(indices.size() == 4, "Indices describe a tetrahedron.");

		std::array<unsigned int, 4> volume = {indices[0], indices[1], indices[2], indices[3]};
		_volumes.push_back(volume);
	}

	void TetraMeshDeserialiser::addNormal(const Vector3f&)
	{
	}
}}}
