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

namespace Vcl { namespace Mathematics {
	int TwoSidedJacobiSVD(Eigen::Matrix<float, 3, 3>& A, Eigen::Matrix<float, 3, 3>& U, Eigen::Matrix<float, 3, 3>& V, bool warm_start);
	int TwoSidedJacobiSVD(Eigen::Matrix<float4, 3, 3>& A, Eigen::Matrix<float4, 3, 3>& U, Eigen::Matrix<float4, 3, 3>& V, bool warm_start);
	int TwoSidedJacobiSVD(Eigen::Matrix<float8, 3, 3>& A, Eigen::Matrix<float8, 3, 3>& U, Eigen::Matrix<float8, 3, 3>& V, bool warm_start);
	int TwoSidedJacobiSVD(Eigen::Matrix<float16, 3, 3>& A, Eigen::Matrix<float16, 3, 3>& U, Eigen::Matrix<float16, 3, 3>& V, bool warm_start);

	inline int TwoSidedJacobiSVD(Eigen::Matrix<float, 3, 3>& A, Eigen::Matrix<float, 3, 3>& U, Eigen::Matrix<float, 3, 3>& V) { return TwoSidedJacobiSVD(A, U, V, false); }
	inline int TwoSidedJacobiSVD(Eigen::Matrix<float4, 3, 3>& A, Eigen::Matrix<float4, 3, 3>& U, Eigen::Matrix<float4, 3, 3>& V) { return TwoSidedJacobiSVD(A, U, V, false); }
	inline int TwoSidedJacobiSVD(Eigen::Matrix<float8, 3, 3>& A, Eigen::Matrix<float8, 3, 3>& U, Eigen::Matrix<float8, 3, 3>& V) { return TwoSidedJacobiSVD(A, U, V, false); }
	inline int TwoSidedJacobiSVD(Eigen::Matrix<float16, 3, 3>& A, Eigen::Matrix<float16, 3, 3>& U, Eigen::Matrix<float16, 3, 3>& V) { return TwoSidedJacobiSVD(A, U, V, false); }

	int TwoSidedJacobiSVD(Eigen::Matrix3d& A, Eigen::Matrix3d& U, Eigen::Matrix3d& V, bool warm_start = false);
}}
