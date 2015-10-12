/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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
#include <vcl/util/mortoncodes.h>

// C++ standard library
#include <random>
#include <vector>

// Google test
#include <gtest/gtest.h>

TEST(MortonCodeTest, Encode)
{
	using namespace Vcl::Util;

	auto c00 = MortonCode::encode(0, 0, 0);
	auto c01 = MortonCode::encode(1, 0, 0);
	auto c02 = MortonCode::encode(0, 1, 0);
	auto c03 = MortonCode::encode(1, 1, 0);
	auto c04 = MortonCode::encode(0, 0, 1);
	auto c05 = MortonCode::encode(1, 0, 1);
	auto c06 = MortonCode::encode(0, 1, 1);
	auto c07 = MortonCode::encode(1, 1, 1);

	EXPECT_EQ(0x0, c00) << "Morton code of (0, 0, 0) is correct";
	EXPECT_EQ(0x1, c01) << "Morton code of (1, 0, 0) is correct";
	EXPECT_EQ(0x2, c02) << "Morton code of (0, 1, 0) is correct";
	EXPECT_EQ(0x3, c03) << "Morton code of (1, 1, 0) is correct";
	EXPECT_EQ(0x4, c04) << "Morton code of (0, 0, 1) is correct";
	EXPECT_EQ(0x5, c05) << "Morton code of (1, 0, 1) is correct";
	EXPECT_EQ(0x6, c06) << "Morton code of (0, 1, 1) is correct";
	EXPECT_EQ(0x7, c07) << "Morton code of (1, 1, 1) is correct";
}

TEST(MortonCodeTest, Decode)
{
	using namespace Vcl::Util;

	std::vector<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>> values =
	{
		{0x0ul, 0, 0, 0},
		{0x1ul, 1, 0, 0},
		{0x2ul, 0, 1, 0},
		{0x3ul, 1, 1, 0},
		{0x4ul, 0, 0, 1},
		{0x5ul, 1, 0, 1},
		{0x6ul, 0, 1, 1},
		{0x7ul, 1, 1, 1}
	};

	for (auto& t : values)
	{
		uint32_t x, y, z;
		MortonCode::decode(std::get<0>(t), x, y, z);

		EXPECT_EQ(x, std::get<1>(t));
		EXPECT_EQ(y, std::get<2>(t));
		EXPECT_EQ(z, std::get<3>(t));
	}
}

TEST(MortonCodeTest, SplitSequence)
{
	using namespace Vcl::Util;

	const std::vector<Eigen::Vector3f> points4x4x4 = []()
	{
		std::vector<Eigen::Vector3f> points;
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 4; j++)
				for (int i = 0; i < 4; i++)
				{
					points.emplace_back(float(i) + 0.25f, float(j) + 0.25f, float(k) + 0.25f);
				}
		return points;
	}();

	std::vector<std::pair<uint64_t, int>> codes;
	codes.resize(points4x4x4.size());
	for (int i = 0; i < codes.size(); i++)
	{
		const Eigen::Vector3f p = points4x4x4[i];
		codes[i] = std::make_pair(MortonCode::encode(p.x(), p.y(), p.z()), i);
	}
	std::sort(codes.begin(), codes.end(), [](const auto& a, const auto& b)
	{
		return a.first < b.first;
	});

	const size_t first_0 = 0;
	const size_t last_0 = codes.size() - 1;
	const auto split_0 = splitHighestBit<int>(codes, first_0, last_0);
	EXPECT_EQ(split_0, (first_0 + last_0) / 2);

	const size_t first_1 = first_0;
	const size_t last_1 = last_0;
	const auto split_1 = splitHighestBit<int>(codes, first_1, last_1);
	EXPECT_EQ(split_1, (first_1 + last_1) / 2);
}
