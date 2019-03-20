/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#include <vcl/geometry/distance_ray3ray3.h>

// VCL
#include <vcl/core/simd/memory.h>

namespace Vcl { namespace Geometry
{
	template<typename Real>
	Real distanceImpl
	(
		const Ray<Real, 3>& ray_a,
		const Ray<Real, 3>& ray_b,
		Result<Real>* result
	)
	{
		// Ray Ray(s) = A + s*P
		// Ray Ray(t) = B + t*Q
		//
		// Distance R(s, t)
		//   = ||Ray(s) - Ray(t)||^2
		//   = ||(A + s*P) - (B + t*Q)||^2
		//   = ||(A - B) + s*P - t*Q||^2
		//       -------   ---   ---
		//          a       b     c
		//   = a^2 + 2ab - 2ac + b^2 - 2bc + c^2
		// 0 = (a^2 - d) + 2ab - 2ac + b^2 - 2bc + c^2
		// dR/ds =  2(A - B)P - 2tPQ + 2sP^2 = 0
		// dR/dt = -2(A - B)Q - 2sPQ + 2tQ^2 = 0
		// | P^2 -PQ | | s | = | -(A - B)P |
		// | -PQ Q^2 | | t | = |  (A - B)Q |
		const auto& A = ray_a.origin();
		const auto& B = ray_b.origin();
		
		const auto& P = ray_a.direction();
		const auto& Q = ray_b.direction();
		const Real pp =  P.dot(P);
		const Real qq =  Q.dot(Q);
		const Real pq = -P.dot(Q);
		const Real c1 = -(A - B).dot(P);
		const Real c2 =  (A - B).dot(Q);

		const Real D  = pp*qq - pq*pq;
		const Real Ds = c1*qq - pq*c2;
		const Real Dt = pp*c2 - c1*pq;

		// Handle parallel rays (D == 0)
		const Real s = select(D > 0, Ds / D, Real(0.0f));
		const Real t = select(D > 0, Dt / D, Real(0.0f));

		const auto& pt_on_a = ray_a(s);
		const auto& pt_on_b = ray_b(t);
		const Real dist = (pt_on_a - pt_on_b).norm();

		if (result)
		{
			result->Parameter[0] = s;
			result->Parameter[1] = t;
			result->Point[0] = pt_on_a;
			result->Point[1] = pt_on_b;
		}

		return dist;
	}

	float distance
	(
		const Ray<float, 3>& ray_a,
		const Ray<float, 3>& ray_b,
		Result<float>* result
	)
	{
		return distanceImpl(ray_a, ray_b, result);
	}
	float4 distance
	(
		const Ray<float4, 3>& ray_a,
		const Ray<float4, 3>& ray_b,
		Result<float4>* result
	)
	{
		return distanceImpl(ray_a, ray_b, result);
	}
	float8 distance
	(
		const Ray<float8, 3>& ray_a,
		const Ray<float8, 3>& ray_b,
		Result<float8>* result
	)
	{
		return distanceImpl(ray_a, ray_b, result);
	}
	float16 distance
	(
		const Ray<float16, 3>& ray_a,
		const Ray<float16, 3>& ray_b,
		Result<float16>* result
	)
	{
		return distanceImpl(ray_a, ray_b, result);
	}
}}
