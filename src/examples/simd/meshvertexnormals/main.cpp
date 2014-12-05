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
#include <vcl/core/simd/memory.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/util/precisetimer.h>

#include "liver.h"

void accumulateNormals
(
	std::vector<Eigen::Vector3i>::const_iterator f_begin,
	std::vector<Eigen::Vector3i>::const_iterator f_end,
	const std::vector<Eigen::Vector3f>& points,
	std::vector<Eigen::Vector3f>& normals
)
{
#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (auto f_itr = f_begin; f_itr != f_end; ++f_itr)
	{
		Eigen::Vector3f p0 = points[f_itr->x()];
		Eigen::Vector3f p1 = points[f_itr->y()];
		Eigen::Vector3f p2 = points[f_itr->z()];

		// Compute the edges
		Eigen::Vector3f p0p1 = p1 - p0; float p0p1_l = p0p1.norm();
		Eigen::Vector3f p1p2 = p2 - p1; float p1p2_l = p1p2.norm();
		Eigen::Vector3f p2p0 = p0 - p2; float p2p0_l = p2p0.norm();

		// Compute the angles
		Eigen::Vector3f angles;

		// Use the dot product between edges: cos t = a.dot(b) / (a.length() * b.length())
		/*angle at v0 */ angles.x() = std::acos((p2 - p0).dot(p1 - p0) / (p2p0_l * p0p1_l));
		/*angle at v1 */ angles.y() = std::acos((p0 - p1).dot(p2 - p1) / (p0p1_l * p1p2_l));
		/*angle at v2 */ angles.z() = std::acos((p1 - p2).dot(p0 - p2) / (p1p2_l * p2p0_l));

		// Compute the normalized face normal
		Eigen::Vector3f n = (p1 - p0).cross(p2 - p0).normalized();

		normals[f_itr->x()] += angles.x() * n;
		normals[f_itr->y()] += angles.y() * n;
		normals[f_itr->z()] += angles.z() * n;
	}
}

template<int Width>
void simdAccumulateNormals
(
	const std::vector<Eigen::Vector3i>& faces,
	const std::vector<Eigen::Vector3f>& points,
	std::vector<Eigen::Vector3f>& normals
)
{
	using Vcl::acos;
	using Vcl::gather;
	using Vcl::load;

	using wint_t = Vcl::VectorScalar<int, Width>;
	using wfloat_t = Vcl::VectorScalar<float, Width>;

	using vector3i_t = Eigen::Matrix<wint_t, 3, 1>;
	using vector3_t = Eigen::Matrix<wfloat_t, 3, 1>;
	
#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < static_cast<int>(faces.size() / Width); i++)
	{
		vector3i_t idx;
		load(idx, faces.data() + Width * i);
		
		vector3_t p0 = gather<float, Width, 3, 1>(points.data(), idx(0), 1);
		vector3_t p1 = gather<float, Width, 3, 1>(points.data(), idx(1), 1);
		vector3_t p2 = gather<float, Width, 3, 1>(points.data(), idx(2), 1);

		// Compute the edges
		vector3_t p0p1 = p1 - p0; wfloat_t p0p1_l = p0p1.norm();
		vector3_t p1p2 = p2 - p1; wfloat_t p1p2_l = p1p2.norm();
		vector3_t p2p0 = p0 - p2; wfloat_t p2p0_l = p2p0.norm();

		// Compute the angles
		vector3_t angles;

		// Use the dot product between edges: cos t = a.dot(b) / (a.length() * b.length())
		/*angle at v0 */ angles.x() = acos((p2 - p0).dot(p1 - p0) / (p2p0_l * p0p1_l));
		/*angle at v1 */ angles.y() = acos((p0 - p1).dot(p2 - p1) / (p0p1_l * p1p2_l));
		/*angle at v2 */ angles.z() = acos((p1 - p2).dot(p0 - p2) / (p1p2_l * p2p0_l));

		// Compute the normalized face normal
		vector3_t n = (p1 - p0).cross(p2 - p0).normalized();
		vector3_t n0 = angles.x() * n;
		vector3_t n1 = angles.y() * n;
		vector3_t n2 = angles.z() * n;

		for (int j = 0; j < Width; j++)
		{
			normals[idx(0)[j]] += Vcl::Vector3f{ n0.x()[j], n0.y()[j], n0.z()[j] };
			normals[idx(1)[j]] += Vcl::Vector3f{ n1.x()[j], n1.y()[j], n1.z()[j] };
			normals[idx(2)[j]] += Vcl::Vector3f{ n2.x()[j], n2.y()[j], n2.z()[j] };
		}
	}
}

void computeNormals
(
	const std::vector<Eigen::Vector3i>& faces,
	const std::vector<Eigen::Vector3f>& points,
	std::vector<Eigen::Vector3f>& normals
)
{
	Vcl::Util::PreciseTimer timer;
	timer.start();

	accumulateNormals(std::begin(faces), std::end(faces), points, normals);

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < (int) points.size(); i++)
	{
		normals[i].normalize();
	}
	timer.stop();
	std::cout << "Compute mesh vertex normals (Reference): " << timer.interval() / faces.size() * 1e9 << "[ns]" << std::endl;
}

template<int Width>
void computeNormalsSIMD
(
	const std::vector<Eigen::Vector3i>& faces,
	const std::vector<Eigen::Vector3f>& points,
	std::vector<Eigen::Vector3f>& normals
)
{
	using Vcl::load;
	using Vcl::store;

	using wint_t = Vcl::VectorScalar<int, Width>;
	using wfloat_t = Vcl::VectorScalar<float, Width>;

	using vector3_t = Eigen::Matrix<wfloat_t, 3, 1>;

	Vcl::Util::PreciseTimer timer;
	timer.start();
	simdAccumulateNormals<Width>(faces, points, normals);

	accumulateNormals(std::begin(faces) + static_cast<int>(faces.size() / Width)*Width, std::end(faces), points, normals);

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < static_cast<int>(points.size() / Width); i++)
	{
		vector3_t n;
		load(n, normals.data() + i*Width);

		// Compute
		n.normalize();

		store(normals.data() + i*Width, n);
	}

	for
	(
		int i = static_cast<int>(points.size() / Width) * Width;
		i < (int) points.size();
		i++
	)
	{
		normals[i].normalize();
	}
	timer.stop();
	std::cout << "Compute mesh vertex normals (SIMD width " << Width << "): " << timer.interval() / faces.size() * 1e9 << "[ns]" << std::endl;

}

int main(int argc, char* argv[])
{
	VCL_UNREFERENCED_PARAMETER(argc);
	VCL_UNREFERENCED_PARAMETER(argv);
	
	// Data set
	std::vector<Eigen::Vector3f> points((Eigen::Vector3f*) liver_points, (Eigen::Vector3f*) (liver_points + num_liver_points));
	std::vector<Eigen::Vector3i> faces((Eigen::Vector3i*) liver_faces, (Eigen::Vector3i*) (liver_faces + num_liver_faces));
	for (auto& idx : faces)
	{
		idx -= Eigen::Vector3i::Ones();
	}

	// Computation output
	std::vector<Eigen::Vector3f> normals_ref(points.size(), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> normals_004(points.size(), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> normals_008(points.size(), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> normals_016(points.size(), Eigen::Vector3f::Zero());

	// Test Performance: Use scalar version
	computeNormals(faces, points, normals_ref);

	// Test Performance: Use paralle versions
	computeNormalsSIMD< 4>(faces, points, normals_004);
	computeNormalsSIMD< 8>(faces, points, normals_008);
	computeNormalsSIMD<16>(faces, points, normals_016);
	
	float L1_004 = 0;
	float L1_008 = 0;
	float L1_016 = 0;
	for (size_t i = 0; i < normals_ref.size(); i++)
	{
		L1_004 += (normals_ref[i] - normals_004[i]).norm();
		L1_008 += (normals_ref[i] - normals_008[i]).norm();
		L1_016 += (normals_ref[i] - normals_016[i]).norm();
	}
	std::cout << "Average error: " << std::endl;
	std::cout << "* SIMD  4: " << L1_004 / normals_ref.size() << std::endl;
	std::cout << "* SIMD  8: " << L1_008 / normals_ref.size() << std::endl;
	std::cout << "* SIMD 16: " << L1_016 / normals_ref.size() << std::endl;

	return 0;
}
