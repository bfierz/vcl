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

// C++ standard library
#include <fstream>
#include <iostream>
#include <random>

// Eigen library
#include <Eigen/Dense>

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/interleavedarray.h>
#include <vcl/math/jacobieigen33_selfadjoint.h>
#include <vcl/math/jacobieigen33_selfadjoint_quat.h>
#include <vcl/util/precisetimer.h>

#include "problems.h"

template<typename WideScalar>
void jacobiEig(
	size_t nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS)
{
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	int avg_nr_iter = 0;
	for (size_t i = 0; i < nr_problems / width; i++)
	{
		// Map data
		auto U = resU.at<real_t>(i);
		auto S = resS.at<real_t>(i);

		// Compute SVD using 2-sided Jacobi iterations (Brent)
		matrix3_t SV = F.at<real_t>(i);
		matrix3_t matU = matrix3_t::Identity();

		avg_nr_iter += Vcl::Mathematics::SelfAdjointJacobiEigen(SV, matU);

		// Store results
		U = matU;
		S = SV.diagonal();
	}
}

template<typename WideScalar>
void jacobiEigQuat(
	size_t nr_problems,
	const Vcl::Core::InterleavedArray<float, 3, 3, -1>& F,
	Vcl::Core::InterleavedArray<float, 3, 3, -1>& resU,
	Vcl::Core::InterleavedArray<float, 3, 1, -1>& resS)
{
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;

	size_t width = sizeof(real_t) / sizeof(float);

	int avg_nr_iter = 0;
	for (size_t i = 0; i < nr_problems / width; i++)
	{
		// Map data
		auto U = resU.at<real_t>(i);
		auto S = resS.at<real_t>(i);

		// Compute SVD using Jacobi iterations and QR decomposition
		matrix3_t SV = F.at<real_t>(i);
		matrix3_t matU = matrix3_t::Identity();

		avg_nr_iter += Vcl::Mathematics::SelfAdjointJacobiEigenQuat(SV, matU);

		// Store results
		U = matU;
		S = SV.diagonal();
	}
}

template<typename REAL>
void SortEigenvalues(Eigen::Matrix<REAL, 3, 1>& A, Eigen::Matrix<REAL, 3, 3>& B)
{
	// Bubble sort
	bool swapped = true;
	int j = 0;

	while (swapped)
	{
		swapped = false;
		j++;

		for (int i = 0; i < 3 - j; i++)
		{
			if (A(i) > A(i + 1))
			{
				std::swap(A(i), A(i + 1));
				B.col(i).swap(B.col(i + 1));

				swapped = true;
			}
		}
	}
}

template<typename Scalar>
void checkSolution(
	const char* Name,
	const char* file,
	size_t nr_problems,
	Scalar tol,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& F,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refUa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& refSa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resUa,
	const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& resSa)
{
	using scalar_t = Scalar;

	int wrong_computations, wrong_u_computations;
	scalar_t accum_error, accum_u_error;
	std::ofstream fout;

	wrong_computations = 0;
	wrong_u_computations = 0;
	accum_error = 0;
	accum_u_error = 0;
	fout.open(file);

	for (size_t j = 0; j < nr_problems; j++)
	{
		Vcl::Matrix3f A = F.template at<scalar_t>(j);
		Vcl::Matrix3f refU = refUa.template at<scalar_t>(j);
		Vcl::Vector3f refS = refSa.template at<scalar_t>(j);
		Vcl::Matrix3f cU = resUa.template at<scalar_t>(j);
		Vcl::Vector3f cS = resSa.template at<scalar_t>(j);

		SortEigenvalues(cS, cU);

		bool eqU = refU.array().abs().isApprox(cU.array().abs(), tol);
		bool eqS = refS.array().abs().isApprox(cS.array().abs(), tol);

		if (!eqS || !eqU)
			fout << j;

		if (!eqS)
		{
			wrong_computations++;
			scalar_t err = abs((refS.array().abs() - cS.array().abs())).maxCoeff();
			accum_error += err;
			fout << ", E: " << err;
		}
		if (!eqU)
		{
			wrong_u_computations++;
			scalar_t err = abs((refU.array().abs() - cU.array().abs())).maxCoeff();
			accum_u_error += err;
			fout << ", U: " << err;
		}
		if (!eqS || !eqU)
			fout << std::endl;
	}

	fout.close();
	std::cout << Name << " - Errors: (" << wrong_computations << ", " << wrong_u_computations << "), "
			  << "Avg. Singular value error: " << accum_error / std::max(wrong_computations, 1) << ", "
			  << "Avg. U error: " << accum_u_error / std::max(wrong_u_computations, 1) << std::endl;
}

int main(int, char**)
{
	size_t nr_problems = 1024 * 1024;

	using scalar_t = float;

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> refS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> F(nr_problems);

	createSymmetricProblems(nr_problems, F);
	computeEigenReferenceSolution(nr_problems, F, refU, refS);

	// Test correctness: Two-sided Jacobi SVD (Brent)
	jacobiEig<float>(nr_problems, F, resU, resS);
	checkSolution("JacobiEigen - float", "jacobi_eigen_float_errors.txt", nr_problems, 1e-5f, F, refU, refS, resU, resS);
	jacobiEig<Vcl::float4>(nr_problems, F, resU, resS);
	checkSolution("JacobiEigen - float4", "jacobi_eigen_float4_errors.txt", nr_problems, 1e-5f, F, refU, refS, resU, resS);
	jacobiEig<Vcl::float8>(nr_problems, F, resU, resS);
	checkSolution("JacobiEigen - float8", "jacobi_eigen_float8_errors.txt", nr_problems, 1e-5f, F, refU, refS, resU, resS);
	jacobiEig<Vcl::float16>(nr_problems, F, resU, resS);
	checkSolution("JacobiEigen - float16", "jacobi_eigen_float16_errors.txt", nr_problems, 1e-5f, F, refU, refS, resU, resS);

	// Test correctness: Jacobi SVD with symmetric EV computation and QR decomposition
	jacobiEigQuat<float>(nr_problems, F, resU, resS);
	checkSolution("JacobiEigenQuat - float", "jacobi_eigen_quat_float_errors.txt", nr_problems, 1e-5f, F, refU, refS, resU, resS);
	jacobiEigQuat<Vcl::float8>(nr_problems, F, resU, resS);
	checkSolution("JacobiEigenQuat - float8", "jacobi_eigen_quat_float8_errors.txt", nr_problems, 1e-5f, F, refU, refS, resU, resS);
	jacobiEigQuat<Vcl::float4>(nr_problems, F, resU, resS);
	checkSolution("JacobiEigenQuat - float4", "jacobi_eigen_quat_float4_errors.txt", nr_problems, 1e-5f, F, refU, refS, resU, resS);
	jacobiEigQuat<Vcl::float16>(nr_problems, F, resU, resS);
	checkSolution("JacobiEigenQuat - float16", "jacobi_eigen_quat_float16_errors.txt", nr_problems, 1e-5f, F, refU, refS, resU, resS);
}
