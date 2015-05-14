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

#include "pitbull.h"

// Tangent space computation is based on 
// Lengyel, Eric. "Computing Tangent Space Basis Vectors for an Arbitrary Mesh".
// Terathon Software 3D Graphics Library, 2001. http://www.terathon.com/code/tangent.html


void accumulateNormals
(
	std::vector<Eigen::Vector3i>::const_iterator f_begin,
	std::vector<Eigen::Vector3i>::const_iterator f_end,
	std::vector<Eigen::Vector3i>::const_iterator tf_begin,
	std::vector<Eigen::Vector3i>::const_iterator tf_end,
	const std::vector<Eigen::Vector3f>& points,
	const std::vector<Eigen::Vector2f>& texcoords,
	std::vector<Eigen::Vector3f>& normals,
	std::vector<Eigen::Vector3f>& tangents,
	std::vector<Eigen::Vector3f>& bitangents
)
{
	VCL_UNREFERENCED_PARAMETER(tf_end);

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (auto f_itr = f_begin; f_itr != f_end; ++f_itr, ++tf_begin)
	{
		Check(tf_begin != tf_end, "Size of faces and texture faces match.");

		Eigen::Vector3f p0 = points[f_itr->x()];
		Eigen::Vector3f p1 = points[f_itr->y()];
		Eigen::Vector3f p2 = points[f_itr->z()];

		// Compute the edges
		Eigen::Vector3f p0p1 = p1 - p0; float p0p1_l = p0p1.norm();
		Eigen::Vector3f p1p2 = p2 - p1; float p1p2_l = p1p2.norm();
		Eigen::Vector3f p2p0 = p0 - p2; float p2p0_l = p2p0.norm();

		// Normalize the edges
		p0p1 = p0p1_l > 1e-6f ? p0p1.normalized() : Eigen::Vector3f::Zero();
		p1p2 = p1p2_l > 1e-6f ? p1p2.normalized() : Eigen::Vector3f::Zero();
		p2p0 = p2p0_l > 1e-6f ? p2p0.normalized() : Eigen::Vector3f::Zero();

		// Compute the angles
		Eigen::Vector3f angles;

		// Use the dot product between edges: cos t = a.dot(b) / (a.length() * b.length())
		/*angle at v0 */ angles.x() = std::acos((-p2p0).dot(p0p1));
		/*angle at v1 */ angles.y() = std::acos((-p0p1).dot(p1p2));
		/*angle at v2 */ angles.z() = std::acos((-p1p2).dot(p2p0));

		// Compute the normalized face normal
		Eigen::Vector3f n = p0p1.cross(-p2p0);

		normals[f_itr->x()] += angles.x() * n;
		normals[f_itr->y()] += angles.y() * n;
		normals[f_itr->z()] += angles.z() * n;

		// Tangent / bitangent
		Eigen::Vector2f w1 = texcoords[tf_begin->x()];
		Eigen::Vector2f w2 = texcoords[tf_begin->y()];
		Eigen::Vector2f w3 = texcoords[tf_begin->z()];

		float x1 =  p0p1.x();
		float x2 = -p2p0.x();
		float y1 =  p0p1.y();
		float y2 = -p2p0.y();
		float z1 =  p0p1.z();
		float z2 = -p2p0.z();

		float s1 = w2.x() - w1.x();
		float s2 = w3.x() - w1.x();
		float t1 = w2.y() - w1.y();
		float t2 = w3.y() - w1.y();

		float r = 1.0F / (s1 * t2 - s2 * t1);
		Eigen::Vector3f sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
		Eigen::Vector3f tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

		tangents[tf_begin->x()] += angles.x() * sdir;
		tangents[tf_begin->y()] += angles.y() * sdir;
		tangents[tf_begin->z()] += angles.z() * sdir;

		bitangents[tf_begin->x()] += angles.x() * tdir;
		bitangents[tf_begin->y()] += angles.y() * tdir;
		bitangents[tf_begin->z()] += angles.z() * tdir;
	}
}

template<int Width>
void simdAccumulateNormals
(
	const std::vector<Eigen::Vector3i>& faces,
	const std::vector<Eigen::Vector3i>& tex_faces,
	const std::vector<Eigen::Vector3f>& points,
	const std::vector<Eigen::Vector2f>& texcoords,
	std::vector<Eigen::Vector3f>& normals,
	std::vector<Eigen::Vector3f>& tangents,
	std::vector<Eigen::Vector3f>& bitangents
)
{
	using Vcl::acos;
	using Vcl::gather;
	using Vcl::load;
	using Vcl::select;

	using wint_t = Vcl::VectorScalar<int, Width>;
	using wfloat_t = Vcl::VectorScalar<float, Width>;

	using vector3i_t = Eigen::Matrix<wint_t, 3, 1>;
	using vector2_t = Eigen::Matrix<wfloat_t, 2, 1>;
	using vector3_t = Eigen::Matrix<wfloat_t, 3, 1>;
	
#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < static_cast<int>(faces.size() / Width); i++)
	{
		vector3i_t idx;
		load(idx, faces.data() + Width * i);
		
		vector3_t p0 = gather<float, Width, 3, 1>(points.data(), idx(0));
		vector3_t p1 = gather<float, Width, 3, 1>(points.data(), idx(1));
		vector3_t p2 = gather<float, Width, 3, 1>(points.data(), idx(2));

		// Compute the edges
		vector3_t p0p1 = p1 - p0; wfloat_t p0p1_l = p0p1.norm();
		vector3_t p1p2 = p2 - p1; wfloat_t p1p2_l = p1p2.norm();
		vector3_t p2p0 = p0 - p2; wfloat_t p2p0_l = p2p0.norm();

		// Normalize the edges
		p0p1 = select<float, Width, 3, 1>(p0p1_l > wfloat_t(1e-6f), p0p1.normalized(), vector3_t::Zero());
		p1p2 = select<float, Width, 3, 1>(p1p2_l > wfloat_t(1e-6f), p1p2.normalized(), vector3_t::Zero());
		p2p0 = select<float, Width, 3, 1>(p2p0_l > wfloat_t(1e-6f), p2p0.normalized(), vector3_t::Zero());

		// Compute the angles
		vector3_t angles;

		// Use the dot product between edges: cos t = a.dot(b) / (a.length() * b.length())
		/*angle at v0 */ angles.x() = acos((-p2p0).dot(p0p1));
		/*angle at v1 */ angles.y() = acos((-p0p1).dot(p1p2));
		/*angle at v2 */ angles.z() = acos((-p1p2).dot(p2p0));

		// Compute the normalized face normal
		vector3_t n = p0p1.cross(-p2p0);
		vector3_t n0 = angles.x() * n;
		vector3_t n1 = angles.y() * n;
		vector3_t n2 = angles.z() * n;

		for (int j = 0; j < Width; j++)
		{
			normals[idx(0)[j]] += Vcl::Vector3f{ n0.x()[j], n0.y()[j], n0.z()[j] };
			normals[idx(1)[j]] += Vcl::Vector3f{ n1.x()[j], n1.y()[j], n1.z()[j] };
			normals[idx(2)[j]] += Vcl::Vector3f{ n2.x()[j], n2.y()[j], n2.z()[j] };
		}

		// Tangent / bitangent
		vector3i_t tidx;
		load(tidx, tex_faces.data() + Width * i);
		vector2_t w1 = gather<float, Width, 2, 1>(texcoords.data(), tidx(0));
		vector2_t w2 = gather<float, Width, 2, 1>(texcoords.data(), tidx(1));
		vector2_t w3 = gather<float, Width, 2, 1>(texcoords.data(), tidx(2));

		wfloat_t x1 = p0p1.x();
		wfloat_t x2 = -p2p0.x();
		wfloat_t y1 = p0p1.y();
		wfloat_t y2 = -p2p0.y();
		wfloat_t z1 = p0p1.z();
		wfloat_t z2 = -p2p0.z();

		wfloat_t s1 = w2.x() - w1.x();
		wfloat_t s2 = w3.x() - w1.x();
		wfloat_t t1 = w2.y() - w1.y();
		wfloat_t t2 = w3.y() - w1.y();

		wfloat_t r = 1.0F / (s1 * t2 - s2 * t1);
		vector3_t sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
		vector3_t tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

		vector3_t sdir0 = angles.x() * sdir;
		vector3_t sdir1 = angles.y() * sdir;
		vector3_t sdir2 = angles.z() * sdir;
		vector3_t tdir0 = angles.x() * tdir;
		vector3_t tdir1 = angles.y() * tdir;
		vector3_t tdir2 = angles.z() * tdir;

		for (int j = 0; j < Width; j++)
		{
			tangents[tidx(0)[j]] += Vcl::Vector3f{ sdir0.x()[j], sdir0.y()[j], sdir0.z()[j] };
			tangents[tidx(1)[j]] += Vcl::Vector3f{ sdir1.x()[j], sdir1.y()[j], sdir1.z()[j] };
			tangents[tidx(2)[j]] += Vcl::Vector3f{ sdir2.x()[j], sdir2.y()[j], sdir2.z()[j] };

			bitangents[tidx(0)[j]] += Vcl::Vector3f{ tdir0.x()[j], tdir0.y()[j], tdir0.z()[j] };
			bitangents[tidx(1)[j]] += Vcl::Vector3f{ tdir1.x()[j], tdir1.y()[j], tdir1.z()[j] };
			bitangents[tidx(2)[j]] += Vcl::Vector3f{ tdir2.x()[j], tdir2.y()[j], tdir2.z()[j] };
		}
	}
}

void computeNormals
(
	const std::vector<Eigen::Vector3i>& faces,
	const std::vector<Eigen::Vector3i>& tex_faces,
	const std::vector<Eigen::Vector3f>& points,
	const std::vector<Eigen::Vector2f>& texcoords,
	std::vector<Eigen::Vector3f>& normals,
	std::vector<Eigen::Vector4f>& out_tangents
)
{
	std::vector<Eigen::Vector3f> tangents(out_tangents.size(), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> bitangents(out_tangents.size(), Eigen::Vector3f::Zero());

	Vcl::Util::PreciseTimer timer;
	timer.start();

	accumulateNormals
	(
		std::begin(faces), std::end(faces),
		std::begin(tex_faces), std::end(tex_faces),
		points, texcoords,
		normals, tangents, bitangents
	);

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < (int) points.size(); i++)
	{
		normals[i].normalize();
	}

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < (int) faces.size(); i++)
	{
		const auto& f = faces[i];
		const auto& tf = tex_faces[i];

		for (int j = 0; j < 3; j++)
		{
			const auto& n = normals[f[j]];
			const auto& t = tangents[tf[j]];

			// Gram-Schmidt orthogonalize
			Eigen::Vector3f tan = (t - n * n.dot(t)).normalized();

			// Calculate handedness
			float hand = (n.cross(t).dot(bitangents[tf[j]]) < 0.0f) ? -1.0f : 1.0f;

			out_tangents[tf[j]] = { tan.x(), tan.y(), tan.z(), hand };
		}
	}

	timer.stop();
	std::cout << "Compute mesh vertex normals (Reference): " << timer.interval() / faces.size() * 1e9 << "[ns]" << std::endl;
}

template<int Width>
void computeNormalsSIMD
(
	const std::vector<Eigen::Vector3i>& faces,
	const std::vector<Eigen::Vector3i>& tex_faces,
	const std::vector<Eigen::Vector3f>& points,
	const std::vector<Eigen::Vector2f>& texcoords,
	std::vector<Eigen::Vector3f>& normals,
	std::vector<Eigen::Vector4f>& out_tangents
)
{
	using Vcl::gather;
	using Vcl::load;
	using Vcl::store;

	using wint_t = Vcl::VectorScalar<int, Width>;
	using wfloat_t = Vcl::VectorScalar<float, Width>;

	using vector3i_t = Eigen::Matrix<wint_t, 3, 1>;
	using vector3f_t = Eigen::Matrix<wfloat_t, 3, 1>;

	std::vector<Eigen::Vector3f> tangents(out_tangents.size(), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> bitangents(out_tangents.size(), Eigen::Vector3f::Zero());

	Vcl::Util::PreciseTimer timer;
	timer.start();
	simdAccumulateNormals<Width>(faces, tex_faces, points, texcoords, normals, tangents, bitangents);

	accumulateNormals
	(
		std::begin(faces) + static_cast<int>(faces.size() / Width)*Width, std::end(faces),
		std::begin(tex_faces) + static_cast<int>(tex_faces.size() / Width)*Width, std::end(tex_faces),
		points, texcoords,
		normals, tangents, bitangents
	);

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < static_cast<int>(points.size() / Width); i++)
	{
		vector3f_t n;
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

#ifdef _OPENMP
#	pragma omp parallel for
#endif // _OPENMP
	for (int i = 0; i < static_cast<int>(faces.size() / Width); i++)
	{
		vector3i_t f, tf;
		load(f, faces.data() + i*Width);
		load(tf, tex_faces.data() + i*Width);

		for (int j = 0; j < 3; j++)
		{
			vector3f_t n = gather<float, Width, 3, 1>(normals.data(), f(j));
			vector3f_t t = gather<float, Width, 3, 1>(tangents.data(), tf(j));
			vector3f_t b = gather<float, Width, 3, 1>(bitangents.data(), tf(j));

			// Gram-Schmidt orthogonalize
			vector3f_t tan = (t - n * n.dot(t)).normalized();

			// Calculate handedness
			wfloat_t hand = select(n.cross(t).dot(b) < 0.0f, wfloat_t{ -1.0f }, wfloat_t{ 1.0f });

			for (int k = 0; k < Width; k++)
			{
				out_tangents[tf(j)[k]] = { tan.x()[k], tan.y()[k], tan.z()[k], hand[k] };
			}
		}
	}

	for (int i = static_cast<int>(faces.size() / Width) * Width; i < (int) faces.size(); i++)
	{
		const auto& f = faces[i];
		const auto& tf = tex_faces[i];

		for (int j = 0; j < 3; j++)
		{
			const auto& n = normals[f[j]];
			const auto& t = tangents[tf[j]];

			// Gram-Schmidt orthogonalize
			Eigen::Vector3f tan = (t - n * n.dot(t)).normalized();

			// Calculate handedness
			float hand = (n.cross(t).dot(bitangents[tf[j]]) < 0.0f) ? -1.0f : 1.0f;

			out_tangents[tf[j]] = { tan.x(), tan.y(), tan.z(), hand };
		}
	}

	timer.stop();
	std::cout << "Compute mesh vertex normals (SIMD width " << Width << "): " << timer.interval() / faces.size() * 1e9 << "[ns]" << std::endl;

}

int main(int argc, char* argv[])
{
	VCL_UNREFERENCED_PARAMETER(argc);
	VCL_UNREFERENCED_PARAMETER(argv);
	
	// Data set
	std::vector<Eigen::Vector3f> points((Eigen::Vector3f*) pitbull_core_points, (Eigen::Vector3f*) (pitbull_core_points + num_pitbull_core_points));
	std::vector<Eigen::Vector2f> texcoords((Eigen::Vector2f*) pitbull_core_texcoords, (Eigen::Vector2f*) (pitbull_core_texcoords + num_pitbull_core_texcoords));
	std::vector<Eigen::Vector3i> faces, tex_faces;

	int num_faces = num_pitbull_core_faces / 9;
	faces.reserve(num_faces);
	tex_faces.reserve(num_faces);
	for (int i = 0; i < num_faces; i++)
	{
		int i00 = pitbull_core_faces[9 * i + 0] - 1;
		int i01 = pitbull_core_faces[9 * i + 1] - 1;
		//int i02 = pitbull_core_faces[9 * i + 2] - 1;

		int i10 = pitbull_core_faces[9 * i + 3] - 1;
		int i11 = pitbull_core_faces[9 * i + 4] - 1;
		//int i12 = pitbull_core_faces[9 * i + 5] - 1;

		int i20 = pitbull_core_faces[9 * i + 6] - 1;
		int i21 = pitbull_core_faces[9 * i + 7] - 1;
		//int i22 = pitbull_core_faces[9 * i + 8] - 1;

		faces.emplace_back(i00, i10, i20);
		tex_faces.emplace_back(i01, i11, i21);
	}

	// Computation output
	std::vector<Eigen::Vector3f> normals_ref(points.size(), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> normals_004(points.size(), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> normals_008(points.size(), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> normals_016(points.size(), Eigen::Vector3f::Zero());

	std::vector<Eigen::Vector4f> tangents_ref(texcoords.size(), Eigen::Vector4f::Zero());
	std::vector<Eigen::Vector4f> tangents_004(texcoords.size(), Eigen::Vector4f::Zero());
	std::vector<Eigen::Vector4f> tangents_008(texcoords.size(), Eigen::Vector4f::Zero());
	std::vector<Eigen::Vector4f> tangents_016(texcoords.size(), Eigen::Vector4f::Zero());

	// Test Performance: Use scalar version
	computeNormals(faces, tex_faces, points, texcoords, normals_ref, tangents_ref);

	// Test Performance: Use paralle versions
	computeNormalsSIMD< 4>(faces, tex_faces, points, texcoords, normals_004, tangents_004);
	computeNormalsSIMD< 8>(faces, tex_faces, points, texcoords, normals_008, tangents_008);
	computeNormalsSIMD<16>(faces, tex_faces, points, texcoords, normals_016, tangents_016);
	
	float L1_004 = 0;
	float L1_008 = 0;
	float L1_016 = 0;
	for (size_t i = 0; i < normals_ref.size(); i++)
	{
		L1_004 += (normals_ref[i] - normals_004[i]).norm();
		L1_008 += (normals_ref[i] - normals_008[i]).norm();
		L1_016 += (normals_ref[i] - normals_016[i]).norm();
	}
	std::cout << "Average error (normals): " << std::endl;
	std::cout << "* SIMD  4: " << L1_004 / normals_ref.size() << std::endl;
	std::cout << "* SIMD  8: " << L1_008 / normals_ref.size() << std::endl;
	std::cout << "* SIMD 16: " << L1_016 / normals_ref.size() << std::endl;

	L1_004 = 0;
	L1_008 = 0;
	L1_016 = 0;
	for (size_t i = 0; i < tangents_ref.size(); i++)
	{
		L1_004 += (tangents_ref[i].segment<3>(0) - tangents_004[i].segment<3>(0)).norm();
		L1_008 += (tangents_ref[i].segment<3>(0) - tangents_008[i].segment<3>(0)).norm();
		L1_016 += (tangents_ref[i].segment<3>(0) - tangents_016[i].segment<3>(0)).norm();
	}
	std::cout << "Average error (tangents): " << std::endl;
	std::cout << "* SIMD  4: " << L1_004 / tangents_ref.size() << std::endl;
	std::cout << "* SIMD  8: " << L1_008 / tangents_ref.size() << std::endl;
	std::cout << "* SIMD 16: " << L1_016 / tangents_ref.size() << std::endl;

	return 0;
}
