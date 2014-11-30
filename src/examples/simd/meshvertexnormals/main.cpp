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
#include <omp.h>
#endif // _OPENMP

// VCL
#include <vcl/core/simd/floatn.h>
#include <vcl/core/simd/memory.h>
#include <vcl/util/precisetimer.h>

#include "liver.h"

int main(int argc, char* argv[])
{
	VCL_UNREFERENCED_PARAMETER(argc);
	VCL_UNREFERENCED_PARAMETER(argv);

	using Vcl::gather;

	//typedef Vcl::float16 real_t;
	typedef Vcl::float8 real_t;
	//typedef Vcl::float4 real_t;
	//typedef float real_t;

	typedef Eigen::Matrix<real_t, 3, 1> vector3_t;

	size_t width = sizeof(real_t) / sizeof(float);

	// Computation output
	std::vector<Eigen::Vector3f> normals(liver_points.size(), Eigen::Vector3f::Zero());

	// Shared objects
	Vcl::Util::PreciseTimer timer;

	// Test Performance: Use scalar version
	timer.start();
	normals.assign(liver_points.size(), Eigen::Vector3f::Zero());

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < liver_faces.size(); i++)
	{
		Eigen::Vector3f p0 = liver_points[liver_faces[i].x() - 1];
		Eigen::Vector3f p1 = liver_points[liver_faces[i].y() - 1];
		Eigen::Vector3f p2 = liver_points[liver_faces[i].z() - 1];
		Eigen::Vector3f n = (p1 - p0).cross(p2 - p0).normalized();

		normals[liver_faces[i].x() - 1] += n;
		normals[liver_faces[i].y() - 1] += n;
		normals[liver_faces[i].z() - 1] += n;
	}

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < (int) liver_points.size(); i++)
	{
		normals[i].normalize();
	}
	timer.stop();
	std::cout << "Compute mesh vertex normals (Reference): " << timer.interval() / liver_faces.size() * 1e9 << "[ns]" << std::endl;

	// Test Performance: Use paralle version
	timer.start();
	normals.assign(liver_points.size(), Eigen::Vector3f::Zero());

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < static_cast<int>(liver_faces.size() / width); i++)
	{
		Vcl::VectorScalar<int, 8> i0, i1, i2;
		for (int i = 0; i < width; i++)
		{
			i0[i] = liver_faces[i].x() - 1;
			i1[i] = liver_faces[i].y() - 1;
			i2[i] = liver_faces[i].z() - 1;
		}

		vector3_t p0 = gather<float, 8, 3, 1>(liver_points.data(), i0, 1);
		vector3_t p1 = gather<float, 8, 3, 1>(liver_points.data(), i1, 1);
		vector3_t p2 = gather<float, 8, 3, 1>(liver_points.data(), i2, 1);

		vector3_t n = (p1 - p0).cross(p2 - p0).normalized();

		for (int i = 0; i < width; i++)
		{
			Vcl::Vector3f tmp{ n.x()[i], n.y()[i], n.z()[i] };

			normals[i0[i]] += tmp;
			normals[i1[i]] += tmp;
			normals[i2[i]] += tmp;
		}
	}

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < static_cast<int>(liver_points.size()); i++)
	{
		normals[i].normalize();
	}
	timer.stop();
	std::cout << "Compute mesh vertex normals (Optimized): " << timer.interval() / liver_faces.size() * 1e9 << "[ns]" << std::endl;

	return 0;
}
