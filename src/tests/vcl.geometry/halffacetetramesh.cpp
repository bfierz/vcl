/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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
#include <vcl/geometry/meshfactory.h>
#include <vcl/geometry/halffacetetramesh.h>

// Google test
#include <gtest/gtest.h>

TEST(HalfFaceTetraMeshTest, SimpleConstruction)
{
	using namespace Vcl::Geometry;

	auto simple_cube = MeshFactory<HalfFaceTetraMesh>::createHomogenousCubes(1, 1, 1);
	
	unsigned int nr_connected_hf = 0;
	unsigned int nr_unconnected_hf = 0;
	for (auto hf_it = simple_cube->halfFaceEnumerator(); !hf_it.empty(); ++hf_it)
	{
		if (hf_it->Opposite.isValid())
			nr_connected_hf++;
		else
			nr_unconnected_hf++;
	}

	EXPECT_EQ(nr_connected_hf, 8);
	EXPECT_EQ(nr_unconnected_hf, 12);

	for (auto hf_it = simple_cube->halfFaceEnumerator(); !hf_it.empty(); ++hf_it)
	{
		const auto op_hf = hf_it->Opposite;
		if (op_hf.isValid())
		{
			auto verts    = simple_cube->verticesFromHalfFace(*hf_it);
			auto op_verts = simple_cube->verticesFromHalfFace(op_hf);

			std::sort(verts.begin(), verts.end());
			std::sort(op_verts.begin(), op_verts.end());

			EXPECT_EQ(verts, op_verts);
		}
	}
}
