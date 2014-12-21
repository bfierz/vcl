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
#include <vcl/math/qr33.h>

// VCL
#include <vcl/math/qr33_impl.h>

namespace Vcl { namespace Mathematics
{
	void jacobiQRDecomposition(Matrix3f& R, Matrix3f& Q)
	{
		// Initialize Q
		Q.setIdentity();

		// Clear values below the diagonal with a fixed sequence (1,0), (2,0), (2,1)
		// of rotations
		jacobiRotateQR<float, 1, 0>(R, Q);
		jacobiRotateQR<float, 2, 0>(R, Q);
		jacobiRotateQR<float, 2, 1>(R, Q);
	}

	void jacobiQRDecomposition(Matrix3d& R, Matrix3d& Q)
	{
		// Initialize Q
		Q.setIdentity();

		// Clear values below the diagonal with a fixed sequence (1,0), (2,0), (2,1)
		// of rotations
		jacobiRotateQR<double, 1, 0>(R, Q);
		jacobiRotateQR<double, 2, 0>(R, Q);
		jacobiRotateQR<double, 2, 1>(R, Q);
	}
}}
