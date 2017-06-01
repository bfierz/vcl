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
#include <vcl/geometry/halfedgetrimesh.h>

namespace Vcl { namespace Geometry
{
	HalfEdgeTriMesh::HalfEdgeTriMesh()
	{
		// Add position data
		_positions = addVertexProperty<Eigen::Vector3f>("Position", Eigen::Vector3f{ 0.0f, 0.0f, 0.0f });
	}

	HalfEdgeTriMesh::HalfEdgeTriMesh(const std::vector<Eigen::Vector3f>& vertices, const std::vector<std::array<IndexDescriptionTrait<HalfEdgeTriMesh>::IndexType, 3>>& faces)
	: HalfEdgeTriMesh()
	{
		addVertices(vertices);

		faceProperties().resizeProperties(faces.size());
		for (size_t i = 0; i < faces.size(); ++i)
		{
			Face f
			{
				VertexId{ faces[i][0] },
				VertexId{ faces[i][1] },
				VertexId{ faces[i][2] }
			};
			_faces[i] = f;
		}

		buildIndex();
	}

	void HalfEdgeTriMesh::clear()
	{
		vertexProperties().clear();
		edgeProperties().clear();
		faceProperties().clear();
	}

	void HalfEdgeTriMesh::addVertices(gsl::span<const Eigen::Vector3f> vertices)
	{
		const size_t curr_size = vertexProperties().propertySize();
		vertexProperties().resizeProperties(curr_size + vertices.size());
		
		for (size_t i = 0; i < vertices.size(); ++i)
		{
			_positions[curr_size + i] = vertices[i];
		}
	}

	void HalfEdgeTriMesh::addFace(const std::array<IndexDescriptionTrait<HalfEdgeTriMesh>::IndexType, 3> face)
	{

	}

	void HalfEdgeTriMesh::buildIndex()
	{
		connectHalfEdges();
	}

	void HalfEdgeTriMesh::connectHalfEdges()
	{

	}
}}
