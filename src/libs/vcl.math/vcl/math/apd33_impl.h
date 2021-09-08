/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <vcl/core/contract.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Mathematics { namespace Impl {
#ifdef VCL_COMPILER_MSVC
#	pragma strict_gs_check(push, off)
#endif // VCL_COMPILER_MSVC

	template<typename Scalar>
	VCL_STRONG_INLINE unsigned int AnalyticPolarDecomposition(const Eigen::Matrix<Scalar, 3, 3>& A, Eigen::Quaternion<Scalar>& q, unsigned int max_iter = 20)
	{
		VclRequire(max_iter > 0, "At least one iteration is requested");

		using matrix3_t = Eigen::Matrix<Scalar, 3, 3>;
		using vector3_t = Eigen::Matrix<Scalar, 3, 1>;

		unsigned int iter = 0;
		vector3_t omega = vector3_t::Identity();
		while (any(omega.norm() > Scalar(0.001)) && iter < max_iter)
		{
			matrix3_t R = q.toRotationMatrix();
			matrix3_t B = R.transpose() * A;

			vector3_t grad{
				B.col(2)[1] - B.col(1)[2],
				B.col(0)[2] - B.col(2)[0],
				B.col(1)[0] - B.col(0)[1]
			};

			Scalar h00 = B.col(1)[1] + B.col(2)[2];
			Scalar h11 = B.col(0)[0] + B.col(2)[2];
			Scalar h22 = B.col(0)[0] + B.col(1)[1];
			Scalar h01 = Scalar(-0.5) * (B.col(1)[0] + B.col(0)[1]);
			Scalar h02 = Scalar(-0.5) * (B.col(2)[0] + B.col(0)[2]);
			Scalar h12 = Scalar(-0.5) * (B.col(2)[1] + B.col(1)[2]);

			Scalar detH = Scalar(-1.0) * h02 * h02 * h11 + Scalar(2.0) * h01 * h02 * h12 - h00 * h12 * h12 - h01 * h01 * h22 + h00 * h11 * h22;

			const Scalar factor = Scalar(-0.25) / detH;
			omega[0] = (h11 * h22 - h12 * h12) * grad[0] + (h02 * h12 - h01 * h22) * grad[1] + (h01 * h12 - h02 * h11) * grad[2];
			omega[0] *= factor;

			omega[1] = (h02 * h12 - h01 * h22) * grad[0] + (h00 * h22 - h02 * h02) * grad[1] + (h01 * h02 - h00 * h12) * grad[2];
			omega[1] *= factor;

			omega[2] = (h01 * h12 - h02 * h11) * grad[0] + (h01 * h02 - h00 * h12) * grad[1] + (h00 * h11 - h01 * h01) * grad[2];
			omega[2] *= factor;

			omega[0] = select(abs(detH) < Scalar(1.0e-9f), -grad[0], omega[0]);
			omega[1] = select(abs(detH) < Scalar(1.0e-9f), -grad[1], omega[1]);
			omega[2] = select(abs(detH) < Scalar(1.0e-9f), -grad[2], omega[2]);

			Scalar useGD = select(omega.dot(grad) > Scalar(0.0), Scalar(1.0), Scalar(-1.0));
			omega[0] = select(useGD > Scalar(0.0), grad[0] * Scalar(-0.125), omega[0]);
			omega[1] = select(useGD > Scalar(0.0), grad[1] * Scalar(-0.125), omega[1]);
			omega[2] = select(useGD > Scalar(0.0), grad[2] * Scalar(-0.125), omega[2]);

			Scalar l_omega2 = omega.squaredNorm();
			const Scalar w = (Scalar(1.0) - l_omega2) / (Scalar(1.0) + l_omega2);
			const vector3_t vec = omega * (Scalar(2.0) / (Scalar(1.0) + l_omega2));
			Eigen::Quaternion<Scalar> update(w, vec.x(), vec.y(), vec.z());
			q = q * update;
			Scalar n = Scalar(1) / q.norm();
			q.coeffs() *= n;

			iter++;
		}

		return iter;
	}

#ifdef VCL_COMPILER_MSVC
#	pragma strict_gs_check(pop)
#endif // VCL_COMPILER_MSVC
}}}
