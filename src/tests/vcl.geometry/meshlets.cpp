/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2026 Basil Fierz
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

// C++ standard library
#include <algorithm>
#include <array>
#include <vector>

// Include the relevant parts from the library
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/meshlets.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

namespace {
	using Triangle = std::array<uint32_t, 3>;

	//! Rotate the triangle so the smallest index comes first, keeping the winding
	Triangle canonical(Triangle t)
	{
		std::rotate(t.begin(), std::min_element(t.begin(), t.end()), t.end());
		return t;
	}

	class MeshletTest : public ::testing::TestWithParam<Vcl::Geometry::MeshletGenerator>
	{
	protected:
		void SetUp() override
		{
			using namespace Vcl::Geometry;

			const auto torus = TriMeshFactory::createTorus(4.0f, 1.0f, 32, 24);
			for (size_t v = 0; v < torus->nrVertices(); v++)
				Vertices.push_back(torus->vertices()[v]);
			for (size_t f = 0; f < torus->nrFaces(); f++)
				for (const auto idx : torus->faces()[f])
					Indices.push_back(idx.id());
		}

		std::vector<Eigen::Vector3f> Vertices;
		std::vector<uint32_t> Indices;
	};
}

TEST_P(MeshletTest, RespectsMeshletLimits)
{
	using namespace Vcl::Geometry;

	const auto mesh = generateMeshlets(GetParam(), Vertices, Indices);
	ASSERT_FALSE(mesh.Meshlets.empty());
	for (const auto& meshlet : mesh.Meshlets)
	{
		EXPECT_LE(meshlet.VertexCount, 64u);
		EXPECT_LE(meshlet.PrimitiveCount, 124u);
		EXPECT_LE(meshlet.VertexOffset + meshlet.VertexCount, mesh.VertexIndices.size());
		EXPECT_LE(meshlet.PrimitiveOffset + 3 * meshlet.PrimitiveCount, mesh.Primitives.size());
	}
}

TEST_P(MeshletTest, ReproducesTheInputTriangles)
{
	using namespace Vcl::Geometry;

	const auto mesh = generateMeshlets(GetParam(), Vertices, Indices);

	std::vector<Triangle> expected;
	for (size_t i = 0; i < Indices.size(); i += 3)
		expected.push_back(canonical({ Indices[i], Indices[i + 1], Indices[i + 2] }));

	std::vector<Triangle> actual;
	for (const auto& meshlet : mesh.Meshlets)
	{
		for (uint32_t p = 0; p < meshlet.PrimitiveCount; p++)
		{
			Triangle t;
			for (uint32_t c = 0; c < 3; c++)
			{
				const uint32_t local = mesh.Primitives[meshlet.PrimitiveOffset + 3 * p + c];
				ASSERT_LT(local, meshlet.VertexCount);
				t[c] = mesh.VertexIndices[meshlet.VertexOffset + local];
			}
			actual.push_back(canonical(t));
		}
	}

	std::sort(expected.begin(), expected.end());
	std::sort(actual.begin(), actual.end());
	EXPECT_EQ(expected, actual);
}

TEST_P(MeshletTest, EmptyInputHasNoMeshlets)
{
	using namespace Vcl::Geometry;

	const auto mesh = generateMeshlets(GetParam(), Vertices, {});
	EXPECT_TRUE(mesh.Meshlets.empty());
	EXPECT_TRUE(mesh.VertexIndices.empty());
	EXPECT_TRUE(mesh.Primitives.empty());
}

INSTANTIATE_TEST_SUITE_P(
	Generators,
	MeshletTest,
	::testing::Values(Vcl::Geometry::MeshletGenerator::Scan, Vcl::Geometry::MeshletGenerator::SpatialClustering));
