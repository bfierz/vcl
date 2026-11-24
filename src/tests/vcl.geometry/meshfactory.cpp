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

// Include the relevant parts from the library
#include <vcl/geometry/meshfactory.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(TriMeshFactoryTest, TorusHasTwoTrianglesPerQuad)
{
	using namespace Vcl::Geometry;

	const unsigned int nr_radial_segments = 8;
	const unsigned int nr_sides = 6;
	const auto torus = TriMeshFactory::createTorus(4.0f, 1.0f, nr_radial_segments, nr_sides);

	EXPECT_EQ((nr_radial_segments + 1) * (nr_sides + 1), torus->nrVertices());
	EXPECT_EQ(2 * nr_radial_segments * nr_sides, torus->nrFaces());
}

TEST(TriMeshFactoryTest, TorusHasNoDegenerateTriangles)
{
	using namespace Vcl::Geometry;

	const auto torus = TriMeshFactory::createTorus(4.0f, 1.0f, 8, 6);
	const auto faces = torus->faces();
	const auto vertices = torus->vertices();
	for (size_t f = 0; f < torus->nrFaces(); f++)
	{
		const auto& face = faces[f];
		const Eigen::Vector3f& p0 = vertices[face[0]];
		const Eigen::Vector3f& p1 = vertices[face[1]];
		const Eigen::Vector3f& p2 = vertices[face[2]];
		EXPECT_GT((p1 - p0).cross(p2 - p0).norm(), 1e-6f) << "Face " << f << " is degenerate";
	}
}
