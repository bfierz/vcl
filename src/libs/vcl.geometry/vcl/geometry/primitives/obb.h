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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// Eigen
#include <Eigen/Dense>

// VCL
#include <vcl/core/span.h>

namespace Vcl { namespace Geometry
{
	//! Fitting of bounding box based on PhD thesis of Stefan Gottschalk
	//! http://gamma.cs.unc.edu/users/gottschalk/main.pdf
	//! Summarized by James Gregson
	//! http://jamesgregson.blogspot.ch/2011/03/latex-test.html
	template<typename Scalar, int Dim>
	class OrientedBox
	{
	public: // Type aliases
		using real_t = Scalar;
		using vector_t = Eigen::Matrix<Scalar, Dim, 1>;
		using matrix_t = Eigen::Matrix<Scalar, Dim, Dim>;

	public:
		OrientedBox(stdext::span<const vector_t> points)
		{
			constructFromPoints(points);
		}

		const vector_t& center() const { return _center; }

	private: // Construction
		void constructFromPoints(stdext::span<const vector_t> points)
		{
			// Find the mean point
			vector_t mu = vector_t::Zero();
			const real_t size = static_cast<real_t>(points.size());
			const real_t inv_size = 1 / size;
			for (const auto& p : points)
			{
				mu += p * inv_size;
			}

			// Build the co-variance matrix
			real_t cxx = -size * mu.x()*mu.x();
			real_t cxy = -size * mu.x()*mu.y();
			real_t cxz = -size * mu.x()*mu.z();
			real_t cyy = -size * mu.y()*mu.y();
			real_t cyz = -size * mu.y()*mu.z();
			real_t czz = -size * mu.z()*mu.z();
			for (const auto& p : points)
			{
				cxx += p.x()*p.x();
				cxy += p.x()*p.y();
				cxz += p.x()*p.z();
				cyy += p.y()*p.y();
				cyz += p.y()*p.z();
				czz += p.z()*p.z();
			}

			matrix_t C = matrix_t::Zero();
			C(0, 0) = cxx; C(0, 1) = cxy; C(0, 2) = cxz;
			C(1, 0) = cxy; C(1, 1) = cyy; C(1, 2) = cyz;
			C(2, 0) = cxz; C(2, 1) = cyz; C(2, 2) = czz;

			evaluateCovariance(points, C);
		}

		void evaluateCovariance(stdext::span<const vector_t> points, const matrix_t& C)
		{
			Eigen::SelfAdjointEigenSolver<matrix_t> solver;
			solver.compute(C, Eigen::ComputeEigenvectors);

			_orientation = solver.eigenvectors();
			_orientation.col(0).normalize();
			_orientation.col(1).normalize();
			_orientation.col(2).normalize();

			// Transform the points the new basis defined by 'rot'
			// and find the minimum and maximum
			Eigen::AlignedBox<Scalar, Dim> box;
			for (const auto& p : points)
			{
				box.extend(_orientation.transpose() * p);
			}

			_center = box.center();
			_extents = box.sizes() / 2;
		}

	private: // Data
		//! Center of the oriented box
		vector_t _center;

		//! Extent of the oriented box along the defined axes
		vector_t _extents;

		//! Orientation of the box, each column defines one axis
		matrix_t _orientation;
	};
}}
