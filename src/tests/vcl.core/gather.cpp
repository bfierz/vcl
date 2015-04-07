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

// Include the relevant parts from the library
#include <vcl/core/simd/memory.h>
#include <vcl/core/simd/vectorscalar.h>

// Google test
#include <gtest/gtest.h>

// Tests the scalar gather function.
TEST(GatherTest, Scalar)
{
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::float16;

	using Vcl::int4;
	using Vcl::int8;
	using Vcl::int16;

	using Vcl::all;

	// Setup the memory
	float mem[] =
	{
		11.3805f, 6.10116f, 11.6117f,
		11.8436f, 6.2012501f, 12.3314f,
		12.1044f, 17.7034f, 12.2123f,
		12.2649f, 17.577f, 12.3399f,
		10.5683f, 10.9295f, 9.10448f,
		10.3209f, 10.5937f, 9.1880402f,
		10.3329f, 14.9963f, 19.0353f,
		10.0455f, 14.6964f, 19.2502f,
		10.1965f, 11.632f, 11.3976f,
		10.0714f, 11.5502f, 11.5733f
	};

	EXPECT_EQ(mem[ 3], Vcl::gather(mem, 3)) << "Scalar code failed.";

	int4 idx4{ 3, 26, 1, 15 };
	float4 ref4{ mem[ 3], mem[26], mem[ 1], mem[15] };

	EXPECT_TRUE(all(ref4 == Vcl::gather(mem, idx4))) << "4-way code failed.";

	int8 idx8{ 3, 26, 1, 15, 12, 29, 0, 19 };
	float8 ref8
	{
		mem[ 3], mem[26], mem[ 1], mem[15],
		mem[12], mem[29], mem[ 0], mem[19]
	};

	EXPECT_TRUE(all(ref8 == Vcl::gather(mem, idx8))) << "8-way code failed.";

	int16 idx16{ 3, 26, 1, 15, 12, 29, 0, 19, 4, 8, 2, 21, 25, 28, 11, 7 };
	float16 ref16
	{
		mem[ 3], mem[26], mem[ 1], mem[15],
		mem[12], mem[29], mem[ 0], mem[19],
		mem[ 4], mem[ 8], mem[ 2], mem[21],
		mem[25], mem[28], mem[11], mem[ 7]
	};

	EXPECT_TRUE(all(ref16 == Vcl::gather(mem, idx16))) << "16-way code failed.";
}

// Tests the matrix gather function.
TEST(GatherTest, Matrix)
{
	using Vcl::float4;
	using Vcl::float8;
	using Vcl::float16;

	using Vcl::int4;
	using Vcl::int8;
	using Vcl::int16;

	using Vcl::all;

	// Setup the memory
	Eigen::Vector3f mem [] =
	{
		Eigen::Vector3f(11.3805f, 6.10116f, 11.6117f),
		Eigen::Vector3f(11.8436f, 6.2012501f, 12.3314f),
		Eigen::Vector3f(12.1044f, 17.7034f, 12.2123f),
		Eigen::Vector3f(12.2649f, 17.577f, 12.3399f),
		Eigen::Vector3f(10.5683f, 10.9295f, 9.10448f),
		Eigen::Vector3f(10.3209f, 10.5937f, 9.1880402f),
		Eigen::Vector3f(10.3329f, 14.9963f, 19.0353f),
		Eigen::Vector3f(10.0455f, 14.6964f, 19.2502f),
		Eigen::Vector3f(10.1965f, 11.632f, 11.3976f),
		Eigen::Vector3f(10.0714f, 11.5502f, 11.5733f),
		Eigen::Vector3f(14.5118f, 12.7151f, 12.1492f),
		Eigen::Vector3f(14.8073f, 12.6484f, 12.1709f),
		Eigen::Vector3f(9.82892f, 9.5343201f, 9.1212402f),
		Eigen::Vector3f(9.9230499f, 9.7417801f, 9.1887703f),
		Eigen::Vector3f(14.7326f, 8.8948402f, 13.287601f),
		Eigen::Vector3f(14.716299f, 8.9874901f, 13.2664f),
		Eigen::Vector3f(5.2040798f, 15.250501f, 19.2868f),
		Eigen::Vector3f(4.7869202f, 15.2801f, 19.1433f),
		Eigen::Vector3f(19.1418f, 7.4918198f, 16.6606f),
		Eigen::Vector3f(19.0258f, 7.7041f, 17.071899f),
		Eigen::Vector3f(14.6468f, 9.0783501f, 13.195f),
		Eigen::Vector3f(6.0853699f, 18.201f, 8.0324303f),
		Eigen::Vector3f(6.24683f, 18.6397f, 8.5483597f),
		Eigen::Vector3f(6.0786701f, 15.605f, 5.3588402f),
		Eigen::Vector3f(6.1671398f, 16.039101f, 5.5938801f),
		Eigen::Vector3f(15.194501f, 10.1853f, 11.6388f),
		Eigen::Vector3f(15.128301f, 10.5658f, 12.0422f),
		Eigen::Vector3f(4.12616f, 10.6454f, 15.616f),
		Eigen::Vector3f(3.7431301f, 11.1153f, 15.519901f),
		Eigen::Vector3f(18.2603f, 12.7318f, 18.630499f)
	};
	
	EXPECT_EQ(mem[ 3], Vcl::gather(mem, 3)) << "Scalar code failed.";

	int4 idx4{ 3, 26, 1, 15 };
	Eigen::Matrix<float4, 3, 1> ref4
	{
		float4(mem[ 3](0), mem[26](0), mem[ 1](0), mem[15](0)),
		float4(mem[ 3](1), mem[26](1), mem[ 1](1), mem[15](1)),
		float4(mem[ 3](2), mem[26](2), mem[ 1](2), mem[15](2))
	};

	EXPECT_TRUE(all(ref4(0) == Vcl::gather<float, 4, 3, 1>(mem, idx4)(0))) << "4-way code failed.";
	EXPECT_TRUE(all(ref4(1) == Vcl::gather<float, 4, 3, 1>(mem, idx4)(1))) << "4-way code failed.";
	EXPECT_TRUE(all(ref4(2) == Vcl::gather<float, 4, 3, 1>(mem, idx4)(2))) << "4-way code failed.";

	int8 idx8{ 3, 26, 1, 15, 12, 29, 0, 19 };
	Eigen::Matrix<float8, 3, 1> ref8
	{
		float8
		(
			mem[ 3](0), mem[26](0), mem[ 1](0), mem[15](0),
			mem[12](0), mem[29](0), mem[ 0](0), mem[19](0)
		),
		float8
		(
			mem[ 3](1), mem[26](1), mem[ 1](1), mem[15](1),
			mem[12](1), mem[29](1), mem[ 0](1), mem[19](1)
		),
		float8
		(
			mem[ 3](2), mem[26](2), mem[ 1](2), mem[15](2),
			mem[12](2), mem[29](2), mem[ 0](2), mem[19](2)
		)
	};

	EXPECT_TRUE(all(ref8(0) == Vcl::gather<float, 8, 3, 1>(mem, idx8)(0))) << "8-way code failed.";
	EXPECT_TRUE(all(ref8(1) == Vcl::gather<float, 8, 3, 1>(mem, idx8)(1))) << "8-way code failed.";
	EXPECT_TRUE(all(ref8(2) == Vcl::gather<float, 8, 3, 1>(mem, idx8)(2))) << "8-way code failed.";

	int16 idx16{ 3, 26, 1, 15, 12, 29, 0, 19, 4, 8, 2, 21, 25, 28, 11, 7 };
	Eigen::Matrix<float16, 3, 1> ref16
	{
		float16
		(
			mem[ 3](0), mem[26](0), mem[ 1](0), mem[15](0),
			mem[12](0), mem[29](0), mem[ 0](0), mem[19](0),
			mem[ 4](0), mem[ 8](0), mem[ 2](0), mem[21](0),
			mem[25](0), mem[28](0), mem[11](0), mem[ 7](0)
		),
		
		float16
		(
			mem[ 3](1), mem[26](1), mem[ 1](1), mem[15](1),
			mem[12](1), mem[29](1), mem[ 0](1), mem[19](1),
			mem[ 4](1), mem[ 8](1), mem[ 2](1), mem[21](1),
			mem[25](1), mem[28](1), mem[11](1), mem[ 7](1)
		),
		
		float16
		(
			mem[ 3](2), mem[26](2), mem[ 1](2), mem[15](2),
			mem[12](2), mem[29](2), mem[ 0](2), mem[19](2),
			mem[ 4](2), mem[ 8](2), mem[ 2](2), mem[21](2),
			mem[25](2), mem[28](2), mem[11](2), mem[ 7](2)
		)
	};

	EXPECT_TRUE(all(ref16(0) == Vcl::gather<float, 16, 3, 1>(mem, idx16)(0))) << "16-way code failed.";
	EXPECT_TRUE(all(ref16(1) == Vcl::gather<float, 16, 3, 1>(mem, idx16)(1))) << "16-way code failed.";
	EXPECT_TRUE(all(ref16(2) == Vcl::gather<float, 16, 3, 1>(mem, idx16)(2))) << "16-way code failed.";
}
