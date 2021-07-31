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
#include <vcl/math/math.h>
#include <vcl/math/qr33.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

TEST(QR33, SimpleGivens)
{
	using Vcl::Matrix3f;
	using Vcl::Mathematics::equal;
	using Vcl::Mathematics::JacobiQR;

	// Sample values from: http://en.wikipedia.org/wiki/QR_decomposition
	Matrix3f A;
	A << 12, -51, 4, 6, 167, -68, -4, 24, -41;

	Matrix3f Ref;
	Ref << 14, 21, -14, 0, 175, -70, 0, 0, 35;

	Matrix3f Iref = Matrix3f::Identity();

	// Compute decomposition
	Matrix3f R = A;
	Matrix3f Q;
	JacobiQR(R, Q);

	// Check reverse result
	Matrix3f Ares = Q * R;
	Matrix3f I = Q.transpose() * Q;

	EXPECT_TRUE(equal(0, R(1, 0), 1e-4f)) << "R(1, 0) is not computed correctly";
	EXPECT_TRUE(equal(0, R(2, 0), 1e-4f)) << "R(2, 0) is not computed correctly";
	EXPECT_TRUE(equal(0, R(2, 1), 1e-4f)) << "R(2, 1) is not computed correctly";
	EXPECT_TRUE(equal(A, Ares, 1e-4f)) << "Verification A = QR failed";
	EXPECT_TRUE(equal(Iref, I, 1e-4f)) << "Verification I = Q^T*Q failed";
}

TEST(QR33, SimpleHouseholder)
{
	using Vcl::Matrix3f;
	using Vcl::Mathematics::equal;
	using Vcl::Mathematics::HouseholderQR;

	// Sample values from: http://en.wikipedia.org/wiki/QR_decomposition
	Matrix3f A;
	A << 12, -51, 4, 6, 167, -68, -4, 24, -41;

	Matrix3f Ref;
	Ref << 14, 21, -14, 0, 175, -70, 0, 0, 35;

	Matrix3f Iref = Matrix3f::Identity();

	// Compute decomposition
	Matrix3f R = A;
	Matrix3f Q;
	HouseholderQR(R, Q);

	// Check reverse result
	Matrix3f Ares = Q * R;
	Matrix3f I = Q.transpose() * Q;

	EXPECT_TRUE(equal(0, R(1, 0), 1e-4f)) << "R(1, 0) is not computed correctly: " << R(1, 0);
	EXPECT_TRUE(equal(0, R(2, 0), 1e-4f)) << "R(2, 0) is not computed correctly: " << R(2, 0);
	EXPECT_TRUE(equal(0, R(2, 1), 1e-4f)) << "R(2, 1) is not computed correctly: " << R(2, 1);
	EXPECT_TRUE(equal(A, Ares, 1e-4f)) << "Verification A = QR failed";
	EXPECT_TRUE(equal(Iref, I, 1e-4f)) << "Verification I = Q^T*Q failed";
}
