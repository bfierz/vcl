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
#include <vcl/geometry/distance.h>

namespace Vcl { namespace Geometry
{
	namespace detail
	{
		template<typename Real>
		VCL_STRONG_INLINE Real inv(const Real& x)
		{
			return Real(1) / x;
		}

		template<typename Real>
		VCL_STRONG_INLINE std::array<Real, 3> computeDistanceRegion0(const Real& s_in, const Real& t_in, const Real& det, const Real& a, const Real& b, const Real& c, const Real& d, const Real& e, const Real& f)
		{
			std::array<Real, 3> dist;

			Real inv_det = inv(det);
			Real s = s_in * inv_det;
			Real t = t_in * inv_det;
			dist[0] = s*(a*s + b*t + ((Real)2.0)*d) +
				      t*(b*s + c*t + ((Real)2.0)*e) + f;
			dist[1] = s;
			dist[2] = t;

			return dist;
		}

		template<typename Real>
		VCL_STRONG_INLINE std::array<Real, 3> computeDistanceRegion1(const Real& det, const Real& a, const Real& b, const Real& c, const Real& d, const Real& e, const Real& f)
		{
			VCL_UNREFERENCED_PARAMETER(det);

			std::array<Real, 3> dist;

			Real numer = c + e - b - d;
			Real denom = a - b*2 + c;

			Real s_a = 0;
			Real s_b = 1;
			Real s_c = numer * inv(denom);

			Real t_a = 1;
			Real t_b = 0;
			Real t_c = (Real)1.0 - s_c;

			Real d_a = c + ((Real)2.0)*e + f;
			Real d_b = a + ((Real)2.0)*d + f;
			Real d_c = s_c*(a*s_c + b*t_c + ((Real)2.0)*d) +
					   t_c*(b*s_c + c*t_c + ((Real)2.0)*e) + f;

			dist[0] = select
			(
				numer <= (Real)0.0,
				d_a,
				select
				(
					numer >= denom,
					d_b,
					d_c
				)
			);
			dist[1] = select
			(
				numer <= (Real)0.0,
				s_a,
				select
				(
					numer >= denom,
					s_b,
					s_c
				)
			);
			dist[2] = select
			(
				numer <= (Real)0.0,
				t_a,
				select
				(
					numer >= denom,
					t_b,
					t_c
				)
			);

			return dist;
		}

		template<typename Real>
		VCL_STRONG_INLINE std::array<Real, 3> computeDistanceRegion2(const Real& det, const Real& a, const Real& b, const Real& c, const Real& d, const Real& e, const Real& f)
		{
			VCL_UNREFERENCED_PARAMETER(det);

			std::array<Real, 3> dist;

			Real tmp0 = b + d;
			Real tmp1 = c + e;
			Real numer = tmp1 - tmp0;
			Real denom = a - b*2 + c;

			Real s_a = 1;
			Real s_b = numer * inv(denom);
			Real s_c = 0;
			Real s_d = 0;
			Real s_e = 0;

			Real t_a = 0;
			Real t_b = (Real)1.0 - s_b;
			Real t_c = 1;
			Real t_d = 0;
			Real t_e = -e * inv(c);

			Real d_a = a + ((Real)2.0)*d + f;
			Real d_b = s_b*(a*s_b + b*t_b + d*2) +
				       t_b*(b*s_b + c*t_b + ((Real)2.0)*e) + f;
			Real d_c = c + ((Real)2.0)*e + f;
			Real d_d = f;
			Real d_e = e*t_e + f;

			dist[0] = select(tmp1 > tmp0,
				select(numer >= denom, d_a, d_b),
				select(tmp1 <= (Real)0.0, d_c,
					select(e >= (Real)0.0, d_d, d_e)));

			dist[1] = select(tmp1 > tmp0,
				select(numer >= denom, s_a, s_b),
				select(tmp1 <= (Real)0.0, s_c,
					select(e >= (Real)0.0, s_d, s_e)));

			dist[2] = select(tmp1 > tmp0,
				select(numer >= denom, t_a, t_b),
				select(tmp1 <= (Real)0.0, t_c,
					select(e >= (Real)0.0, t_d, t_e)));
			return dist;
		}

		template<typename Real>
		VCL_STRONG_INLINE std::array<Real, 3> computeDistanceRegion3(const Real& det, const Real& a, const Real& b, const Real& c, const Real& d, const Real& e, const Real& f)
		{
			VCL_UNREFERENCED_PARAMETER(det);
			VCL_UNREFERENCED_PARAMETER(a);
			VCL_UNREFERENCED_PARAMETER(b);
			VCL_UNREFERENCED_PARAMETER(d);


			std::array<Real, 3> dist;

			Real t_a = 0;
			Real t_b = 1;
			Real t_c = -e * inv(c);

			Real sq_d_a = f;
			Real sq_d_b = c + ((Real)2.0)*e + f;
			Real sq_d_c = e*t_c + f;

			dist[0] = select
			(
				e >= (Real)0.0,
				sq_d_a,
				select
				(
					-e >= c,
					sq_d_b,
					sq_d_c
				)
			);
			dist[1] = 0;
			dist[2] = select
			(
				e >= (Real)0.0,
				t_a,
				select
				(
					-e >= c,
					t_b,
					t_c
				)
			);

			return dist;
		}

		template<typename Real>
		VCL_STRONG_INLINE std::array<Real, 3> computeDistanceRegion4(const Real& det, const Real& a, const Real& b, const Real& c, const Real& d, const Real& e, const Real& f)
		{
			VCL_UNREFERENCED_PARAMETER(det);
			VCL_UNREFERENCED_PARAMETER(b);


			std::array<Real, 3> dist;

			Real s_a = 1;
			Real s_b = -d * inv(a);
			Real s_c = 0;
			Real s_d = 0;
			Real s_e = 0;
					   
			Real t_a = 0;
			Real t_b = 0;
			Real t_c = 0;
			Real t_d = 1;
			Real t_e = -e * inv(c);
					   
			Real d_a = a + ((Real)2.0)*d + f;
			Real d_b = d*s_b + f;
			Real d_c = f;
			Real d_d = c + ((Real)2.0)*e + f;
			Real d_e = e*t_e + f;

			dist[0] = select(d < (Real)0.0, 
				select(-d >= a, d_a, d_b),
				select(e >= (Real)0.0, d_c,
					select(-e >= c, d_d, d_e)));
					
			dist[1] = select(d < (Real)0.0, 
				select(-d >= a, s_a, s_b),
				select(e >= (Real)0.0, s_c,
					select(-e >= c, s_d, s_e)));
					
			dist[2] = select(d < (Real)0.0, 
				select(-d >= a, t_a, t_b),
				select(e >= (Real)0.0, t_c,
					select(-e >= c, t_d, t_e)));

			return dist;
		}

		template<typename Real>
		VCL_STRONG_INLINE std::array<Real, 3> computeDistanceRegion5(const Real& det, const Real& a, const Real& b, const Real& c, const Real& d, const Real& e, const Real& f)
		{
			VCL_UNREFERENCED_PARAMETER(det);
			VCL_UNREFERENCED_PARAMETER(b);
			VCL_UNREFERENCED_PARAMETER(c);
			VCL_UNREFERENCED_PARAMETER(e);


			std::array<Real, 3> dist;

			Real s_a = 0;
			Real s_b = 1;
			Real s_c = -d * inv(a);

			Real d_a = f;
			Real d_b = a + d*2 + f;
			Real d_c = d*s_c + f;

			dist[0] = select(d >= 0,
				d_a,
				select(-d >= a, d_b, d_c));
			dist[1] = select(d >= 0, 
				s_a,
				select(-d >= a, s_b, s_c));
			dist[2] = 0;
			return dist;
		}

		template<typename Real>
		VCL_STRONG_INLINE std::array<Real, 3> computeDistanceRegion6(const Real& det, const Real& a, const Real& b, const Real& c, const Real& d, const Real& e, const Real& f)
		{
			VCL_UNREFERENCED_PARAMETER(det);

			std::array<Real, 3> dist;

			Real tmp0 = b + e;
			Real tmp1 = a + d;
			Real numer = tmp1 - tmp0;
			Real denom = a - ((Real)2.0)*b + c;

			Real t_a = 1;
			Real t_b = numer * inv(denom);
			Real t_c = 0;
			Real t_d = 0;
			Real t_e = 0;

			Real s_a = 0;
			Real s_b = (Real)1.0 - t_b;
			Real s_c = 1;
			Real s_d = 0;
			Real s_e = -d * inv(a);
					   
			Real d_a = c + ((Real)2.0)*e + f;
			Real d_b = s_b*(a*s_b + b*t_b + ((Real)2.0)*d) +
					   t_b*(b*s_b + c*t_b + ((Real)2.0)*e) + f;
			Real d_c = a + ((Real)2.0)*d + f;
			Real d_d = f;
			Real d_e = d*s_e + f;

			dist[0] = select(tmp1 > tmp0,
				select(numer >= denom, d_a, d_b),
				select(tmp1 <= (Real)0.0, d_c,
					select(d >= (Real)0.0, d_d, d_e)));
			
			dist[1] = select(tmp1 > tmp0,
				select(numer >= denom, s_a, s_b),
				select(tmp1 <= (Real)0.0, s_c,
					select(d >= (Real)0.0, s_d, s_e)));

			dist[2] = select(tmp1 > tmp0,
				select(numer >= denom, t_a, t_b),
				select(tmp1 <= (Real)0.0, t_c,
					select(d >= (Real)0.0, t_d, t_e)));

			return dist;
		}
	}

	template<typename Real>
	Real distanceImpl
	(
		const Triangle<Real, 3>& tri,
		const Eigen::Matrix<Real, 3, 1>& p,
		std::array<Real, 3>* barycentric,
		int* r
	)
	{
		using namespace Vcl::Mathematics;

		Eigen::Matrix<Real, 3, 1> P = p;
		Eigen::Matrix<Real, 3, 1> B = tri[0];
		Eigen::Matrix<Real, 3, 1> E0 = tri[1] - tri[0];
		Eigen::Matrix<Real, 3, 1> E1 = tri[2] - tri[0];
		Real a = E0.squaredNorm();
		Real b = E0.dot(E1);
		Real c = E1.squaredNorm();
		Real d = (B - P).dot(E0);
		Real e = (B - P).dot(E1);
		Real f = (B - P).squaredNorm();
		Real det = abs(a*c-b*b);
		Real s = b*e-c*d;
		Real t = b*d-a*e;

		// Compute the results for all the regions
		std::array<Real, 3> sq_dist = select
		(
			s + t <= det, 
			select
			(
				s < (Real)0.0,
				select(t < (Real)0.0, detail::computeDistanceRegion4(det, a, b, c, d, e, f), detail::computeDistanceRegion3(det, a, b, c, d, e, f)),
				select(t < (Real)0.0, detail::computeDistanceRegion5(det, a, b, c, d, e, f), detail::computeDistanceRegion0(s, t, det, a, b, c, d, e, f))
			),
			select
			(
				s < (Real)0.0,
				detail::computeDistanceRegion2(det, a, b, c, d, e, f),
				select(t < (Real)0.0, detail::computeDistanceRegion6(det, a, b, c, d, e, f), detail::computeDistanceRegion1(det, a, b, c, d, e, f))
			)
		);

		//int region = select
		//(
		//	s + t <= det, 
		//	select
		//	(
		//		s < (Real)0.0,
		//		select(t < (Real)0.0, 4, 3), 
		//		select(t < (Real)0.0, 5, 0)
		//	),
		//	select
		//	(
		//		s < (Real)0.0,
		//		2,
		//		select(t < (Real)0.0, 6, 1)
		//	)
		//);

		// Account for numerical round-off error
		sq_dist[0] = max((Real) 0, sq_dist[0]);

		if (barycentric)
		{
			(*barycentric)[0] = (Real)1.0 - s - t;
			(*barycentric)[1] = sq_dist[1];
			(*barycentric)[2] = sq_dist[2];
		}

		//if (r != nullptr)
		//	*r = region;
		if (r != nullptr)
			*r = -1;

		/*m_kClosestPoint0 = P;
		m_kClosestPoint1 = B + s*E0+ t*E1;*/
		return sqrt(sq_dist[0]);
	}

	float   distance(const Triangle<float, 3>& tri, const Eigen::Matrix<float, 3, 1>& p, std::array<float, 3>* barycentric, int* r)
	{
		return distanceImpl(tri, p, barycentric, r);
	}
	float4  distance(const Triangle<float4, 3>& tri, const Eigen::Matrix<float4, 3, 1>& p, std::array<float4, 3>* barycentric, int* r)
	{
		return distanceImpl(tri, p, barycentric, r);
	}
	float8  distance(const Triangle<float8, 3>& tri, const Eigen::Matrix<float8, 3, 1>& p, std::array<float8, 3>* barycentric, int* r)
	{
		return distanceImpl(tri, p, barycentric, r);
	}
	float16 distance(const Triangle<float16, 3>& tri, const Eigen::Matrix<float16, 3, 1>& p, std::array<float16, 3>* barycentric, int* r)
	{
		return distanceImpl(tri, p, barycentric, r);
	}
}}
