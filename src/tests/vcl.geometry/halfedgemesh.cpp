/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ Standard Library

// Include the relevant parts from the library
#include <vcl/core/container/array.h>
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/halfedgemesh.h>

// Google test
#include <gtest/gtest.h>

// Tests the tetra mesh
TEST(HalfEdgeMeshTest, SimpleConstruction)
{
	using namespace Vcl::Geometry;

	constexpr unsigned int side = 4;
	std::vector<Eigen::Vector2f> points;
	points.reserve(side*side);
	std::set<std::array<HalfEdgeMesh::VertexId, 2>> edges;

	for (unsigned int y = 0; y < side; y++)
	{
		for (unsigned int x = 0; x < side; x++)
		{
			points.emplace_back(x, y);

			if (x + 1 < side && y + 1 < side)
			{
				using Id = HalfEdgeMesh::VertexId;
				Id v00{ (y + 0)*side + x + 0 };
				Id v01{ (y + 0)*side + x + 1 };
				Id v10{ (y + 1)*side + x + 0 };
				Id v11{ (y + 1)*side + x + 1 };

				edges.emplace(std::make_array(v00, v01));
				edges.emplace(std::make_array(v01, v11));
				edges.emplace(std::make_array(v11, v10));
				edges.emplace(std::make_array(v10, v00));
				edges.emplace(std::make_array(v01, v10));
			}
		}
	}

	HalfEdgeMesh mesh;
	mesh.addVertices(points);

	for (const auto& e : edges)
		mesh.addEdge(e);

	// Test one-ring
	{
		HalfEdgeMesh::VertexId center{ 5 };
		std::set<HalfEdgeMesh::VertexId> ref_one_ring
		{
			HalfEdgeMesh::VertexId{ 1 },
			HalfEdgeMesh::VertexId{ 2 },
			HalfEdgeMesh::VertexId{ 4 },
			HalfEdgeMesh::VertexId{ 6 },
			HalfEdgeMesh::VertexId{ 8 },
			HalfEdgeMesh::VertexId{ 9 }
		};
		std::set<HalfEdgeMesh::VertexId> one_ring;

		HalfEdgeMesh::HalfEdgeId start_he_id = mesh.vertex(center).HalfEdge;
		HalfEdgeMesh::HalfEdgeId curr_he_id = start_he_id;
		do
		{
			const auto& he = mesh.halfEdge(mesh.halfEdge(curr_he_id).Twin);
			one_ring.emplace(he.Vertex);
			curr_he_id = he.Next;
		} while (curr_he_id != start_he_id);

		EXPECT_EQ(one_ring, ref_one_ring);
	}

	// Test single face
	{
		HalfEdgeMesh::VertexId corner{ 0 };
		std::set<HalfEdgeMesh::VertexId> ref_face
		{
			HalfEdgeMesh::VertexId{ 0 },
			HalfEdgeMesh::VertexId{ 1 },
			HalfEdgeMesh::VertexId{ 4 }
		};
		std::set<HalfEdgeMesh::VertexId> face;

		HalfEdgeMesh::HalfEdgeId start_he_id = mesh.vertex(corner).HalfEdge;
		HalfEdgeMesh::HalfEdgeId curr_he_id = start_he_id;
		do
		{
			const auto& he = mesh.halfEdge(curr_he_id);
			face.emplace(he.Vertex);
			curr_he_id = he.Next;
		} while (curr_he_id != start_he_id);

		EXPECT_EQ(face, ref_face);
	}
}
