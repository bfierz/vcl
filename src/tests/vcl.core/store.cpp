/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2023 Basil Fierz
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

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

// Tests the scalar gather function.
TEST(StoreTest, Scalar)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	using Vcl::int16;
	using Vcl::int4;
	using Vcl::int8;

	using Vcl::all;

	// Setup the memory
	alignas(64) const float mem[] = {
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
	alignas(64) float store[30];

	float f = mem[3];
	Vcl::store(store + 3, f);
	EXPECT_EQ(store[3], mem[3]) << "Scalar code failed.";

	float4 f4{ mem[5], mem[6], mem[7], mem[8] };
	Vcl::store(store + 5, f4);
	EXPECT_EQ(memcmp(mem + 5, store + 5, 4 * sizeof(float)), 0) << "4-way code failed.";

	float8 f8{
		mem[9], mem[10], mem[11], mem[12],
		mem[13], mem[14], mem[15], mem[16]
	};
	Vcl::store(store + 9, f8);
	EXPECT_EQ(memcmp(mem + 9, store + 9, 8 * sizeof(float)), 0) << "8-way code failed.";

	float16 f16{
		mem[7], mem[8], mem[9], mem[10],
		mem[11], mem[12], mem[13], mem[14],
		mem[15], mem[16], mem[17], mem[18],
		mem[19], mem[20], mem[21], mem[22]
	};
	Vcl::store(store + 7, f16);
	EXPECT_EQ(memcmp(mem + 7, store + 7, 16 * sizeof(float)), 0) << "16-way code failed.";
}

// Tests the matrix gather function.
TEST(StoreTest, Vector2)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	using Vcl::int16;
	using Vcl::int4;
	using Vcl::int8;

	using Vcl::all;

	// Setup the memory
	alignas(64) const Eigen::Vector2f mem[] = {
		Eigen::Vector2f(11.3805f, 6.10116f),
		Eigen::Vector2f(11.8436f, 6.2012501f),
		Eigen::Vector2f(12.1044f, 17.7034f),
		Eigen::Vector2f(12.2649f, 17.577f),
		Eigen::Vector2f(10.5683f, 10.9295f),
		Eigen::Vector2f(10.3209f, 10.5937f),
		Eigen::Vector2f(10.3329f, 14.9963f),
		Eigen::Vector2f(10.0455f, 14.6964f),
		Eigen::Vector2f(10.1965f, 11.632f),
		Eigen::Vector2f(10.0714f, 11.5502f),
		Eigen::Vector2f(14.5118f, 12.7151f),
		Eigen::Vector2f(14.8073f, 12.6484f),
		Eigen::Vector2f(9.82892f, 9.5343201f),
		Eigen::Vector2f(9.9230499f, 9.7417801f),
		Eigen::Vector2f(14.7326f, 8.8948402f),
		Eigen::Vector2f(14.716299f, 8.9874901f),
		Eigen::Vector2f(5.2040798f, 15.250501f),
		Eigen::Vector2f(4.7869202f, 15.2801f),
		Eigen::Vector2f(19.1418f, 7.4918198f),
		Eigen::Vector2f(19.0258f, 7.7041f),
		Eigen::Vector2f(14.6468f, 9.0783501f),
		Eigen::Vector2f(6.0853699f, 18.201f),
		Eigen::Vector2f(6.24683f, 18.6397f),
		Eigen::Vector2f(6.0786701f, 15.605f),
		Eigen::Vector2f(6.1671398f, 16.039101f),
		Eigen::Vector2f(15.194501f, 10.1853f),
		Eigen::Vector2f(15.128301f, 10.5658f),
		Eigen::Vector2f(4.12616f, 10.6454f),
		Eigen::Vector2f(3.7431301f, 11.1153f),
		Eigen::Vector2f(18.2603f, 12.7318f)
	};
	alignas(64) Eigen::Vector2f store[30];

	Eigen::Vector2f f = mem[3];
	Vcl::store(store + 3, f);
	EXPECT_EQ(memcmp(mem + 3, store + 3, 1 * sizeof(Eigen::Vector2f)), 0) << "Scalar code failed.";

	Eigen::Matrix<float4, 2, 1> f4{
		float4(mem[5](0), mem[6](0), mem[7](0), mem[8](0)),
		float4(mem[5](1), mem[6](1), mem[7](1), mem[8](1))
	};
	Vcl::store(store + 5, f4);
	EXPECT_EQ(memcmp(mem + 5, store + 5, 4 * sizeof(Eigen::Vector2f)), 0) << "4-way code failed.";

	Eigen::Matrix<float8, 2, 1> f8{
		float8(
			mem[9](0), mem[10](0), mem[11](0), mem[12](0),
			mem[13](0), mem[14](0), mem[15](0), mem[16](0)),
		float8(
			mem[9](1), mem[10](1), mem[11](1), mem[12](1),
			mem[13](1), mem[14](1), mem[15](1), mem[16](1))
	};
	Vcl::store(store + 9, f8);
	EXPECT_EQ(memcmp(mem + 9, store + 9, 8 * sizeof(Eigen::Vector2f)), 0) << "8-way code failed.";

	Eigen::Matrix<float16, 2, 1> f16{
		float16(
			mem[7](0), mem[8](0), mem[9](0), mem[10](0),
			mem[11](0), mem[12](0), mem[13](0), mem[14](0),
			mem[15](0), mem[16](0), mem[17](0), mem[18](0),
			mem[19](0), mem[20](0), mem[21](0), mem[22](0)),

		float16(
			mem[7](1), mem[8](1), mem[9](1), mem[10](1),
			mem[11](1), mem[12](1), mem[13](1), mem[14](1),
			mem[15](1), mem[16](1), mem[17](1), mem[18](1),
			mem[19](1), mem[20](1), mem[21](1), mem[22](1))
	};
	Vcl::store(store + 7, f16);
	EXPECT_EQ(memcmp(mem + 7, store + 7, 16 * sizeof(Eigen::Vector2f)), 0) << "16-way code failed.";
}

TEST(StoreTest, Vector3)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	using Vcl::int16;
	using Vcl::int4;
	using Vcl::int8;

	using Vcl::all;

	// Setup the memory
	alignas(64) const Eigen::Vector3f mem[] = {
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
	alignas(64) Eigen::Vector3f store[30];

	Eigen::Vector3f f = mem[3];
	Vcl::store(store + 3, f);
	EXPECT_EQ(memcmp(mem + 3, store + 3, 1 * sizeof(Eigen::Vector3f)), 0) << "Scalar code failed.";

	Eigen::Matrix<float4, 3, 1> f4{
		float4(mem[5](0), mem[6](0), mem[7](0), mem[8](0)),
		float4(mem[5](1), mem[6](1), mem[7](1), mem[8](1)),
		float4(mem[5](2), mem[6](2), mem[7](2), mem[8](2))
	};
	Vcl::store(store + 5, f4);
	EXPECT_EQ(memcmp(mem + 5, store + 5, 4 * sizeof(Eigen::Vector3f)), 0) << "4-way code failed.";

	Eigen::Matrix<float8, 3, 1> f8{
		float8(
			mem[9](0), mem[10](0), mem[11](0), mem[12](0),
			mem[13](0), mem[14](0), mem[15](0), mem[16](0)),
		float8(
			mem[9](1), mem[10](1), mem[11](1), mem[12](1),
			mem[13](1), mem[14](1), mem[15](1), mem[16](1)),
		float8(
			mem[9](2), mem[10](2), mem[11](2), mem[12](2),
			mem[13](2), mem[14](2), mem[15](2), mem[16](2))
	};
	Vcl::store(store + 9, f8);
	EXPECT_EQ(memcmp(mem + 9, store + 9, 8 * sizeof(Eigen::Vector3f)), 0) << "8-way code failed.";

	Eigen::Matrix<float16, 3, 1> f16{
		float16(
			mem[7](0), mem[8](0), mem[9](0), mem[10](0),
			mem[11](0), mem[12](0), mem[13](0), mem[14](0),
			mem[15](0), mem[16](0), mem[17](0), mem[18](0),
			mem[19](0), mem[20](0), mem[21](0), mem[22](0)),

		float16(
			mem[7](1), mem[8](1), mem[9](1), mem[10](1),
			mem[11](1), mem[12](1), mem[13](1), mem[14](1),
			mem[15](1), mem[16](1), mem[17](1), mem[18](1),
			mem[19](1), mem[20](1), mem[21](1), mem[22](1)),

		float16(
			mem[7](2), mem[8](2), mem[9](2), mem[10](2),
			mem[11](2), mem[12](2), mem[13](2), mem[14](2),
			mem[15](2), mem[16](2), mem[17](2), mem[18](2),
			mem[19](2), mem[20](2), mem[21](2), mem[22](2))
	};
	Vcl::store(store + 7, f16);
	EXPECT_EQ(memcmp(mem + 7, store + 7, 16 * sizeof(Eigen::Vector3f)), 0) << "16-way code failed.";
}

TEST(StoreTest, Vector4)
{
	using Vcl::float16;
	using Vcl::float4;
	using Vcl::float8;

	using Vcl::int16;
	using Vcl::int4;
	using Vcl::int8;

	using Vcl::all;

	// Setup the memory
	alignas(64) const Eigen::Vector4f mem[] = {
		Eigen::Vector4f(0, 1, 3, 4),
		Eigen::Vector4f(5, 6, 7, 8),
		Eigen::Vector4f(9, 10, 11, 12),
		Eigen::Vector4f(13, 14, 15, 16),
		Eigen::Vector4f(17, 18, 19, 20),
		Eigen::Vector4f(21, 22, 23, 24),
		Eigen::Vector4f(25, 26, 27, 28),
		Eigen::Vector4f(29, 30, 31, 32),
		Eigen::Vector4f(33, 34, 35, 36),
		Eigen::Vector4f(37, 38, 39, 40),
		Eigen::Vector4f(41, 42, 43, 44),
		Eigen::Vector4f(45, 46, 47, 48),
		Eigen::Vector4f(49, 50, 51, 52),
		Eigen::Vector4f(53, 54, 55, 56),
		Eigen::Vector4f(57, 58, 59, 60),
		Eigen::Vector4f(61, 62, 63, 64),
		Eigen::Vector4f(65, 66, 67, 68),
		Eigen::Vector4f(69, 70, 71, 72),
		Eigen::Vector4f(73, 74, 75, 76),
		Eigen::Vector4f(77, 78, 79, 80),
		Eigen::Vector4f(81, 82, 83, 84),
		Eigen::Vector4f(85, 86, 87, 88),
		Eigen::Vector4f(89, 90, 91, 92),
		Eigen::Vector4f(93, 94, 95, 96)
	};
	alignas(64) Eigen::Vector4f store[30];

	Eigen::Vector4f f = mem[3];
	Vcl::store(store + 3, f);
	EXPECT_TRUE(mem[3] == f) << "Scalar code failed.";

	Eigen::Matrix<float4, 4, 1> f4{
		float4(mem[5](0), mem[6](0), mem[7](0), mem[8](0)),
		float4(mem[5](1), mem[6](1), mem[7](1), mem[8](1)),
		float4(mem[5](2), mem[6](2), mem[7](2), mem[8](2)),
		float4(mem[5](3), mem[6](3), mem[7](3), mem[8](3))
	};
	Vcl::store(store + 5, f4);
	EXPECT_EQ(memcmp(mem + 5, store + 5, 4 * sizeof(Eigen::Vector4f)), 0) << "4-way code failed.";

	Eigen::Matrix<float8, 4, 1> ref8{
		float8(
			mem[9](0), mem[10](0), mem[11](0), mem[12](0),
			mem[13](0), mem[14](0), mem[15](0), mem[16](0)),
		float8(
			mem[9](1), mem[10](1), mem[11](1), mem[12](1),
			mem[13](1), mem[14](1), mem[15](1), mem[16](1)),
		float8(
			mem[9](2), mem[10](2), mem[11](2), mem[12](2),
			mem[13](2), mem[14](2), mem[15](2), mem[16](2)),
		float8(
			mem[9](3), mem[10](3), mem[11](3), mem[12](3),
			mem[13](3), mem[14](3), mem[15](3), mem[16](3))
	};

	Eigen::Matrix<float8, 4, 1> f8{
		float8(
			mem[9](0), mem[10](0), mem[11](0), mem[12](0),
			mem[13](0), mem[14](0), mem[15](0), mem[16](0)),
		float8(
			mem[9](1), mem[10](1), mem[11](1), mem[12](1),
			mem[13](1), mem[14](1), mem[15](1), mem[16](1)),
		float8(
			mem[9](2), mem[10](2), mem[11](2), mem[12](2),
			mem[13](2), mem[14](2), mem[15](2), mem[16](2)),
		float8(
			mem[9](3), mem[10](3), mem[11](3), mem[12](3),
			mem[13](3), mem[14](3), mem[15](3), mem[16](3))
	};
	Vcl::store(store + 9, f8);
	EXPECT_EQ(memcmp(mem + 9, store + 9, 8 * sizeof(Eigen::Vector4f)), 0) << "8-way code failed.";

	Eigen::Matrix<float16, 4, 1> f16{
		float16(
			mem[7](0), mem[8](0), mem[9](0), mem[10](0),
			mem[11](0), mem[12](0), mem[13](0), mem[14](0),
			mem[15](0), mem[16](0), mem[17](0), mem[18](0),
			mem[19](0), mem[20](0), mem[21](0), mem[22](0)),

		float16(
			mem[7](1), mem[8](1), mem[9](1), mem[10](1),
			mem[11](1), mem[12](1), mem[13](1), mem[14](1),
			mem[15](1), mem[16](1), mem[17](1), mem[18](1),
			mem[19](1), mem[20](1), mem[21](1), mem[22](1)),

		float16(
			mem[7](2), mem[8](2), mem[9](2), mem[10](2),
			mem[11](2), mem[12](2), mem[13](2), mem[14](2),
			mem[15](2), mem[16](2), mem[17](2), mem[18](2),
			mem[19](2), mem[20](2), mem[21](2), mem[22](2)),

		float16(
			mem[7](3), mem[8](3), mem[9](3), mem[10](3),
			mem[11](3), mem[12](3), mem[13](3), mem[14](3),
			mem[15](3), mem[16](3), mem[17](3), mem[18](3),
			mem[19](3), mem[20](3), mem[21](3), mem[22](3))
	};
	Vcl::store(store + 7, f16);
	EXPECT_EQ(memcmp(mem + 7, store + 7, 16 * sizeof(Eigen::Vector4f)), 0) << "16-way code failed.";
}
