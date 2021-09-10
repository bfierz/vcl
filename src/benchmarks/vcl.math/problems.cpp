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
#include "problems.h"

// C++ standard library
#include <random>

// Eigen library
#include <Eigen/Dense>

// VCL
#include <vcl/math/polardecomposition.h>

void createRandomProblems(
	size_t nr_problems,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>* R)
{
	// Random number generator
	std::mt19937_64 rng;
	std::uniform_real_distribution<float> d;

	for (int i = 0; i < (int)nr_problems; i++)
	{
		// Rest-state
		Eigen::Matrix3f M;
		M << d(rng), d(rng), d(rng),
			d(rng), d(rng), d(rng),
			d(rng), d(rng), d(rng);
		F.at<float>(i) = M;

		if (R)
		{
			Eigen::Matrix3f Rot;
			Vcl::Mathematics::PolarDecomposition(M, Rot, nullptr);
			R->template at<float>(i) = Rot;
		}
	}
}

void createSymmetricProblems(
	size_t nr_problems,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>* R)
{
	// Random number generator
	std::mt19937_64 rng;
	std::uniform_real_distribution<float> d;

	for (int i = 0; i < (int)nr_problems; i++)
	{
		// Rest-state
		Eigen::Matrix3f M;
		M << d(rng), d(rng), d(rng),
			d(rng), d(rng), d(rng),
			d(rng), d(rng), d(rng);
		Eigen::Matrix3f MtM = M.transpose() * M;
		F.at<float>(i) = MtM;

		if (R)
		{
			Eigen::Matrix3f Rot;
			Vcl::Mathematics::PolarDecomposition(MtM, Rot, nullptr);
			R->template at<float>(i) = Rot;
		}
	}
}

void createRotationProblems(
	size_t nr_problems,
	float max_angle,
	float max_compression,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>* R)
{
	// Random number generator
	std::mt19937_64 rng;
	std::uniform_real_distribution<float> d;
	std::uniform_real_distribution<float> a{ -max_angle, max_angle };

	for (int i = 0; i < (int)nr_problems; i++)
	{
		// Rest-state
		Eigen::Matrix3f X0;
		X0 << d(rng), d(rng), d(rng),
			d(rng), d(rng), d(rng),
			d(rng), d(rng), d(rng);

		// Rotation angle
		float angle = a(rng);

		// Rotation axis
		Eigen::Matrix<float, 3, 1> rot_vec;
		rot_vec << d(rng), d(rng), d(rng);
		rot_vec.normalize();

		// Rotation matrix
		Eigen::Matrix3f Rot = Eigen::AngleAxis<float>{ angle, rot_vec }.toRotationMatrix();
		if (R)
			R->template at<float>(i) = Rot;

		if (max_compression > 0)
		{
			Eigen::Matrix<float, 3, 1> scaling;
			scaling << (1.0f - max_compression * d(rng)), (1.0f - max_compression * d(rng)), (1.0f - max_compression * d(rng));

			Eigen::JacobiSVD<Eigen::Matrix3f> svd{ Rot, Eigen::ComputeFullU | Eigen::ComputeFullV };
			Rot *= svd.matrixV() * scaling.asDiagonal() * svd.matrixV().transpose();
		}

		Eigen::Matrix3f X = Rot * X0;
		F.at<float>(i) = X * X0.inverse();
	}
}

void computeEigenReferenceSolution(
	size_t nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& ATA,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& U,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& S)
{
	// Compute reference using Eigen
	for (int i = 0; i < static_cast<int>(nr_problems); i++)
	{
		Vcl::Matrix3f A = ATA.at<float>(i);

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver;
		solver.compute(A, Eigen::ComputeEigenvectors);

		U.at<float>(i) = solver.eigenvectors();
		S.at<float>(i) = solver.eigenvalues();
	}
}
