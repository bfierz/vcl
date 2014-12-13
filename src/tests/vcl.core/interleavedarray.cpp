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
#include <random>
#include <vector>


// Include the relevant parts from the library
#include <vcl/core/interleavedarray.h>

// Google test
#include <gtest/gtest.h>

template<int ROWS, int COLS>
void consecutiveLayoutTest(size_t size)
{
	// Random number generator for the reference data
	std::mt19937 rnd_dev;
	std::uniform_real_distribution<float> rnd_dist;
	auto rnd = std::bind(rnd_dist, rnd_dev);

	// Create reference data
	std::vector<Eigen::Matrix<float, ROWS, COLS>> ref_data(size);
	std::for_each(std::begin(ref_data), std::end(ref_data), [&rnd](Eigen::Matrix<float, ROWS, COLS>& data)
	{
		for (int c = 0; c < data.cols(); c++)
		{
			for (int r = 0; r < data.rows(); r++)
			{
				data(r, c) = rnd();
			}
		}
	});

	// Create the data container to test
	Vcl::Core::InterleavedArray<float, ROWS, COLS, 0> data0(size);
	Vcl::Core::InterleavedArray<float, ROWS, COLS, 1> data1(size);

	// Check the configuration
	EXPECT_EQ(data0.size(), (size_t) size);
	EXPECT_EQ(data1.size(), (size_t) size);

	// Check if the data is written correctly to the storage
	for (int i = 0; i < size; i++)
	{
		data0.at<float>(i) = ref_data[i];
		data1.at<float>(i) = ref_data[i];
	}

	float* data_ptr0 = data0.data();
	float* data_ptr1 = data1.data();
	size_t stride = ROWS*COLS;
	for (int i = 0; i < size; i++)
	{
		size_t base = i*stride;

		bool check0 = true;
		bool check1 = true;
		for (int c = 0; c < COLS; c++)
		{
			for (int r = 0; r < ROWS; r++)
			{
				bool equal0 = data_ptr0[base + c*ROWS + r] == ref_data[i](r, c);
				check0 = check0 && equal0;

				bool equal1 = data_ptr1[base + c*ROWS + r] == ref_data[i](r, c);
				check1 = check1 && equal1;
			}
		}

		EXPECT_TRUE(check0);
		EXPECT_TRUE(check1);
	}

	// Check if the data is read correctly from the storage
	bool check0 = true;
	bool check1 = true;
	for (int i = 0; i < size; i++)
	{
		bool equal0 = data0.at<float>(i) == ref_data[i];
		check0 = check0 && equal0;

		bool equal1 = data1.at<float>(i) == ref_data[i];
		check1 = check1 && equal1;
	}
	EXPECT_TRUE(check0);
	EXPECT_TRUE(check1);

}

template<int ROWS, int COLS, int STRIDE>
void stridedLayoutTest(size_t size)
{
	// Random number generator for the reference data
	std::mt19937 rnd_dev;
	std::uniform_real_distribution<float> rnd_dist;
	auto rnd = std::bind(rnd_dist, rnd_dev);

	// Create reference data
	std::vector<Eigen::Matrix<float, ROWS, COLS>> ref_data(size);
	std::for_each(std::begin(ref_data), std::end(ref_data), [&rnd](Eigen::Matrix<float, ROWS, COLS>& data)
	{
		for (int c = 0; c < data.cols(); c++)
		{
			for (int r = 0; r < data.rows(); r++)
			{
				data(r, c) = rnd();
			}
		}
	});

	// Create the data container to test
	Vcl::Core::InterleavedArray<float, ROWS, COLS, STRIDE> data0(size);
	Vcl::Core::InterleavedArray<float, ROWS, COLS, Vcl::Core::DynamicStride> data1(size, ROWS, COLS, STRIDE);

	// Check the configuration
	EXPECT_EQ(data0.size(), (size_t) size);
	EXPECT_EQ(data1.size(), (size_t) size);

	// Check if the data is written correctly to the storage
	for (int i = 0; i < size; i++)
	{
		data0.at<float>(i) = ref_data[i];
		data1.at<float>(i) = ref_data[i];
	}

	float* data_ptr0 = data0.data();
	float* data_ptr1 = data1.data();
	size_t stride = STRIDE*ROWS*COLS;
	for (int i = 0; i < size; i++)
	{
		size_t base = (i / STRIDE)*stride;
		size_t j = (i / STRIDE) * STRIDE;

		for (int s = 0; s < STRIDE && j + s < size; s++)
		{
			bool check0 = true;
			bool check1 = true;
			for (int c = 0; c < COLS; c++)
			{
				for (int r = 0; r < ROWS; r++)
				{
					bool equal0 = data_ptr0[base + c*ROWS*STRIDE + r*STRIDE + s] == ref_data[j + s](r, c);
					check0 = check0 && equal0;

					bool equal1 = data_ptr1[base + c*ROWS*STRIDE + r*STRIDE + s] == ref_data[j + s](r, c);
					check1 = check1 && equal1;
				}
			}
			EXPECT_TRUE(check0);
			EXPECT_TRUE(check1);
		}
	}

	// Check if the data is read correctly from the storage
	bool check0 = true;
	bool check1 = true;
	for (int i = 0; i < size; i++)
	{
		bool equal0 = data0.at<float>(i) == ref_data[i];
		check0 = check0 && equal0;

		bool equal1 = data1.at<float>(i) == ref_data[i];
		check1 = check1 && equal1;
	}
	EXPECT_TRUE(check0);
	EXPECT_TRUE(check1);
}

template<int STRIDE>
void stridedLayoutTestStub()
{
	stridedLayoutTest<3, 1, STRIDE>(1);
	stridedLayoutTest<3, 1, STRIDE>(31);
	stridedLayoutTest<3, 1, STRIDE>(32);
	stridedLayoutTest<3, 1, STRIDE>(33);

	stridedLayoutTest<1, 3, STRIDE>(1);
	stridedLayoutTest<1, 3, STRIDE>(31);
	stridedLayoutTest<1, 3, STRIDE>(32);
	stridedLayoutTest<1, 3, STRIDE>(33);

	stridedLayoutTest<3, 3, STRIDE>(1);
	stridedLayoutTest<3, 3, STRIDE>(31);
	stridedLayoutTest<3, 3, STRIDE>(32);
	stridedLayoutTest<3, 3, STRIDE>(33);

	stridedLayoutTest<3, 4, STRIDE>(1);
	stridedLayoutTest<3, 4, STRIDE>(31);
	stridedLayoutTest<3, 4, STRIDE>(32);
	stridedLayoutTest<3, 4, STRIDE>(33);

	stridedLayoutTest<4, 3, STRIDE>(1);
	stridedLayoutTest<4, 3, STRIDE>(31);
	stridedLayoutTest<4, 3, STRIDE>(32);
	stridedLayoutTest<4, 3, STRIDE>(33);
}

TEST(InterleavedArrayTest, consecutiveLayout)
{
	consecutiveLayoutTest<3, 1>(1);
	consecutiveLayoutTest<3, 1>(31);
	consecutiveLayoutTest<3, 1>(32);
	consecutiveLayoutTest<3, 1>(33);

	consecutiveLayoutTest<1, 3>(1);
	consecutiveLayoutTest<1, 3>(31);
	consecutiveLayoutTest<1, 3>(32);
	consecutiveLayoutTest<1, 3>(33);

	consecutiveLayoutTest<3, 3>(1);
	consecutiveLayoutTest<3, 3>(31);
	consecutiveLayoutTest<3, 3>(32);
	consecutiveLayoutTest<3, 3>(33);

	consecutiveLayoutTest<3, 4>(1);
	consecutiveLayoutTest<3, 4>(31);
	consecutiveLayoutTest<3, 4>(32);
	consecutiveLayoutTest<3, 4>(33);

	consecutiveLayoutTest<4, 3>(1);
	consecutiveLayoutTest<4, 3>(31);
	consecutiveLayoutTest<4, 3>(32);
	consecutiveLayoutTest<4, 3>(33);
}

TEST(InterleavedArrayTest, stridedLayout2) { stridedLayoutTestStub<2>(); }
TEST(InterleavedArrayTest, stridedLayout3) { stridedLayoutTestStub<3>(); }
TEST(InterleavedArrayTest, stridedLayout4) { stridedLayoutTestStub<4>(); }
TEST(InterleavedArrayTest, stridedLayout7) { stridedLayoutTestStub<7>(); }
TEST(InterleavedArrayTest, stridedLayout8) { stridedLayoutTestStub<8>(); }
TEST(InterleavedArrayTest, stridedLayout31) { stridedLayoutTestStub<31>(); }
TEST(InterleavedArrayTest, stridedLayout32) { stridedLayoutTestStub<32>(); }
TEST(InterleavedArrayTest, stridedLayout33) { stridedLayoutTestStub<33>(); }
