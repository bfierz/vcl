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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// VCL
#include <vcl/core/simd/vectorscalar.h>

namespace Vcl { namespace Mathematics
{
	/*
	 *	Original implementation by:
	 *		2011 - McAdams, Selle, Tamstorf, Teran, Sifakis - Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
	 *	which is an extensive description of the method presented in
	 *		SIGGRAPH - 2011 - McAdams, Zhu, Selle, Empey, Tamstorf, Teran, Sifakis - Efficient elasticity for character skinning with contact and collisions
	 */
	int McAdamsJacobiSVD(Eigen::Matrix<float , 3, 3>& A, Eigen::Matrix<float , 3, 3>& U, Eigen::Matrix<float , 3, 3>& V);
	int McAdamsJacobiSVD(Eigen::Matrix<float4, 3, 3>& A, Eigen::Matrix<float4, 3, 3>& U, Eigen::Matrix<float4, 3, 3>& V);
	int McAdamsJacobiSVD(Eigen::Matrix<float8, 3, 3>& A, Eigen::Matrix<float8, 3, 3>& U, Eigen::Matrix<float8, 3, 3>& V);
}}
