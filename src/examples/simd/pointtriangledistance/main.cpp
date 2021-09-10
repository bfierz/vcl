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

// C++ standard library
#include <iostream>

// OpenMP
#ifdef _OPENMP
#	include <omp.h>
#endif // _OPENMP

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/geometry/distancePoint3Triangle3.h>
#include <vcl/util/precisetimer.h>

template<typename Real>
Real distanceEberly(
	const Eigen::Matrix<Real, 3, 1>& v0,
	const Eigen::Matrix<Real, 3, 1>& v1,
	const Eigen::Matrix<Real, 3, 1>& v2,
	const Eigen::Matrix<Real, 3, 1>& p,
	std::array<Real, 3>* barycentric,
	int* r);

int main(int argc, char* argv[])
{
	using Vcl::Geometry::distance;

	//using real_t = Vcl::float16;
	using real_t = Vcl::float8;
	//using real_t = Vcl::float4;
	//using real_t = float;

	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	// Reference triangle
	Eigen::Vector3f ref_a{ 1, 0, 0 };
	Eigen::Vector3f ref_b{ 0, 1, 0 };
	Eigen::Vector3f ref_c{ 0, 0, 1 };

	vector3_t a{ 1, 0, 0 };
	vector3_t b{ 0, 1, 0 };
	vector3_t c{ 0, 0, 1 };

	size_t nr_problems = 1024 * 1024;

	std::vector<Eigen::Vector3f> ref_points(nr_problems);
	Vcl::Core::InterleavedArray<float, 3, 1, -1> points(nr_problems);

	// Strides
	size_t width = sizeof(real_t) / sizeof(float);

	// Initialize data
	for (int i = 0; i < (int)nr_problems; i++)
	{
		ref_points[i].setRandom();
		points.at<float>(i) = ref_points[i];
	}

	// Shared objects
	Vcl::Util::PreciseTimer timer;

	// Test Performance: Eberly's reference method
	timer.start();
#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < (int)nr_problems; i++)
	{
		Eigen::Vector3f p = ref_points[i];
		std::array<float, 3> st;
		int r1 = -1;

		float d1 = distanceEberly(ref_a, ref_b, ref_c, p, &st, &r1);
		VCL_UNREFERENCED_PARAMETER(d1);
	}
	timer.stop();
	std::cout << "Point-triangle Distance (Reference): " << timer.interval() / nr_problems * 1e9 << "[ns]" << std::endl;

	// Test Performance: Optimized test by Eberly
	timer.start();
#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < static_cast<int>(nr_problems / width); i++)
	{
		vector3_t p = points.at<real_t>(i);

		std::array<real_t, 3> st;
		int r1 = -1;

		real_t d1 = distance({ a, b, c }, p, &st, &r1);
		VCL_UNREFERENCED_PARAMETER(d1);
	}
	timer.stop();
	std::cout << "Point-triangle Distance (Optimized): " << timer.interval() / nr_problems * 1e9 << "[ns]" << std::endl;

	return 0;
}

template<typename Real>
Real distanceEberly(
	const Eigen::Matrix<Real, 3, 1>& v0,
	const Eigen::Matrix<Real, 3, 1>& v1,
	const Eigen::Matrix<Real, 3, 1>& v2,
	const Eigen::Matrix<Real, 3, 1>& p,
	std::array<Real, 3>* barycentric,
	int* r)
{
	Eigen::Matrix<Real, 3, 1> P = p;
	Eigen::Matrix<Real, 3, 1> B = v0;
	Eigen::Matrix<Real, 3, 1> E0 = v1 - v0;
	Eigen::Matrix<Real, 3, 1> E1 = v2 - v0;
	Real a = E0.squaredNorm();
	Real b = E0.dot(E1);
	Real c = E1.squaredNorm();
	Real d = (B - P).dot(E0);
	Real e = (B - P).dot(E1);
	Real f = (B - P).squaredNorm();
	Real det = std::abs(a * c - b * b);
	Real s = b * e - c * d;
	Real t = b * d - a * e;
	Real dist;

	int region = -1;

	if (s + t <= det)
	{
		if (s < (Real)0.0)
		{
			if (t < (Real)0.0) // region 4
			{
				region = 4;

				if (d < (Real)0.0)
				{
					t = (Real)0.0;
					if (-d >= a)
					{
						s = (Real)1.0;
						dist = a + ((Real)2.0) * d + f;
					} else
					{
						s = -d / a;
						dist = d * s + f;
					}
				} else
				{
					s = (Real)0.0;
					if (e >= (Real)0.0)
					{
						t = (Real)0.0;
						dist = f;
					} else if (-e >= c)
					{
						t = (Real)1.0;
						dist = c + ((Real)2.0) * e + f;
					} else
					{
						t = -e / c;
						dist = e * t + f;
					}
				}
			} else // region 3
			{
				region = 3;

				s = (Real)0.0;
				if (e >= (Real)0.0)
				{
					t = (Real)0.0;
					dist = f;
				} else if (-e >= c)
				{
					t = (Real)1.0;
					dist = c + ((Real)2.0) * e + f;
				} else
				{
					t = -e / c;
					dist = e * t + f;
				}
			}
		} else if (t < (Real)0.0) // region 5
		{
			region = 5;

			t = (Real)0.0;
			if (d >= (Real)0.0)
			{
				s = (Real)0.0;
				dist = f;
			} else if (-d >= a)
			{
				s = (Real)1.0;
				dist = a + ((Real)2.0) * d + f;
			} else
			{
				s = -d / a;
				dist = d * s + f;
			}
		} else // region 0
		{
			region = 0;

			// minimum at interior point
			Real inv_det = ((Real)1.0) / det;
			s *= inv_det;
			t *= inv_det;
			dist = s * (a * s + b * t + ((Real)2.0) * d) +
				   t * (b * s + c * t + ((Real)2.0) * e) + f;
		}
	} else
	{
		Real tmp0, tmp1, numer, denom;

		if (s < (Real)0.0) // region 2
		{
			region = 2;

			tmp0 = b + d;
			tmp1 = c + e;
			if (tmp1 > tmp0)
			{
				numer = tmp1 - tmp0;
				denom = a - 2.0f * b + c;
				if (numer >= denom)
				{
					s = (Real)1.0;
					t = (Real)0.0;
					dist = a + ((Real)2.0) * d + f;
				} else
				{
					s = numer / denom;
					t = (Real)1.0 - s;
					dist = s * (a * s + b * t + 2.0f * d) +
						   t * (b * s + c * t + ((Real)2.0) * e) + f;
				}
			} else
			{
				s = (Real)0.0;
				if (tmp1 <= (Real)0.0)
				{
					t = (Real)1.0;
					dist = c + ((Real)2.0) * e + f;
				} else if (e >= (Real)0.0)
				{
					t = (Real)0.0;
					dist = f;
				} else
				{
					t = -e / c;
					dist = e * t + f;
				}
			}
		} else if (t < (Real)0.0) // region 6
		{
			region = 6;

			tmp0 = b + e;
			tmp1 = a + d;
			if (tmp1 > tmp0)
			{
				numer = tmp1 - tmp0;
				denom = a - ((Real)2.0) * b + c;
				if (numer >= denom)
				{
					t = (Real)1.0;
					s = (Real)0.0;
					dist = c + ((Real)2.0) * e + f;
				} else
				{
					t = numer / denom;
					s = (Real)1.0 - t;
					dist = s * (a * s + b * t + ((Real)2.0) * d) +
						   t * (b * s + c * t + ((Real)2.0) * e) + f;
				}
			} else
			{
				t = (Real)0.0;
				if (tmp1 <= (Real)0.0)
				{
					s = (Real)1.0;
					dist = a + ((Real)2.0) * d + f;
				} else if (d >= (Real)0.0)
				{
					s = (Real)0.0;
					dist = f;
				} else
				{
					s = -d / a;
					dist = d * s + f;
				}
			}
		} else // region 1
		{
			region = 1;

			numer = c + e - b - d;
			if (numer <= (Real)0.0)
			{
				s = (Real)0.0;
				t = (Real)1.0;
				dist = c + ((Real)2.0) * e + f;
			} else
			{
				denom = a - 2.0f * b + c;
				if (numer >= denom)
				{
					s = (Real)1.0;
					t = (Real)0.0;
					dist = a + ((Real)2.0) * d + f;
				} else
				{
					s = numer / denom;
					t = (Real)1.0 - s;
					dist = s * (a * s + b * t + ((Real)2.0) * d) +
						   t * (b * s + c * t + ((Real)2.0) * e) + f;
				}
			}
		}
	}

	// account for numerical round-off error
	if (dist < (Real)0.0)
	{
		dist = (Real)0.0;
	}

	if (barycentric)
	{
		(*barycentric)[0] = (Real)1.0 - s - t;
		(*barycentric)[1] = s;
		(*barycentric)[2] = t;
	}

	if (r != nullptr)
		*r = region;

	/*m_kClosestPoint0 = P;
	m_kClosestPoint1 = B + s*E0+ t*E1;*/
	return std::sqrt(dist);
}
