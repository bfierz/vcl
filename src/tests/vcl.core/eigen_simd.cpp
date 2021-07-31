/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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

// Include the relevant parts from the library
#include <vcl/core/simd/vectorscalar.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(EigenSimdIntegration, BoundingBoxInitialization)
{
	using namespace Vcl;

	Eigen::AlignedBox<Vcl::float4, 2> aabb;
	Eigen::Matrix<Vcl::float4, 2, 1> max = Eigen::Matrix<Vcl::float4, 2, 1>::Constant(std::numeric_limits<float>::max());
	Eigen::Matrix<Vcl::float4, 2, 1> min = Eigen::Matrix<Vcl::float4, 2, 1>::Constant(-std::numeric_limits<float>::max());

	EXPECT_TRUE(Vcl::all(aabb.min().x() == max.x()));
	EXPECT_TRUE(Vcl::all(aabb.min().y() == max.y()));
	EXPECT_TRUE(Vcl::all(aabb.max().x() == min.x()));
	EXPECT_TRUE(Vcl::all(aabb.max().y() == min.y()));
}
