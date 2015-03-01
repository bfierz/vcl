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

// Google test
#include <gtest/gtest.h>

// Tests the scalar gather function.
TEST(QR33, Simple)
{
	using Vcl::Matrix3f;
	using Vcl::Mathematics::equal;
	using Vcl::Mathematics::JacobiQR;

	// Sample values from: http://en.wikipedia.org/wiki/QR_decomposition
	Matrix3f A;
	A << 12, -51, 4, 6, 167, -68, -4, 24, -41;

	Matrix3f Ref;
	Ref << 14, 21, -14, 0, 175, -70, 0, 0, 35;

	// Compute decomposition
	Matrix3f R = A;
	Matrix3f Q;
	JacobiQR(R, Q);

	EXPECT_TRUE(equal(Ref, R, 1e-4f)) << "Simple Example";
}
