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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

namespace Vcl { namespace Geometry
{
	template<typename Scalar, int Dim>
	class Circle
	{
	public:
		using real_t = Scalar;
		using vector_t = Eigen::Matrix<Scalar, Dim, 1>;

	public:
		Circle(const vector_t& center, const vector_t& normal, real_t radius)
		: _center(center)
		, _normal(normal)
		, _radius(radius)
		{
			static_assert(Dim >= 2, "A circle needs at least 2 dimensions.");
		}

		Circle(const vector_t& p0, const vector_t& p1, const vector_t& p2)
		{
		}

	public:
		const vector_t& center() const { return _center; }
		const vector_t& normal() const { return _normal; }
		real_t radius() const { return _radius; }

	private:
		//! Center of the circle
		vector_t _center;

		//! Plane defining the orientation of the circle
		vector_t _normal;

		//! Circle radius
		real_t _radius;
	};

	template<typename Scalar>
	class Circle<Scalar, 2>
	{
	public:
		using real_t = Scalar;
		using vector_t = Eigen::Matrix<Scalar, 2, 1>;

	public:
		Circle(vector_t center, real_t radius)
		: _center(center)
		, _radius(radius)
		{}

		Circle(const vector_t& p0, const vector_t& p1, const vector_t& p2)
		{
			// Compute centers of p0p1 and p1p2
			const vector_t z0 = (p0 + p1) / 2.0f;
			const vector_t z1 = (p1 + p2) / 2.0f;

			const vector_t n0{ p1.y() - p0.y(), -(p1.x() - p0.x()) };
			const vector_t n1{ p2.y() - p1.y(), -(p2.x() - p1.x()) };

			// 1. c0.x + s*n0.x = c1.x + t*n1.x
			// 1. c0.y + s*n0.y = c1.y + t*n1.y
			// 2. (c0.x - c1.x) + s*n0.x = t*n1.x
			// 2. (c0.y - c1.y) + s*n0.y = t*n1.y
			// 3. s*n0.x - t*n1.x = -(c0.x - c1.x)
			// 3. s*n0.y - t*n1.y = -(c0.y - c1.y)
			const real_t a1 = n0.x();
			const real_t a2 = n0.y();
			const real_t b1 = -n1.x();
			const real_t b2 = -n1.y();
			const real_t c1 = -(z0.x() - z1.x());
			const real_t c2 = -(z0.y() - z1.y());

			const real_t D = a1*b2 - b1*a2;
			const real_t Ds = c1*b2 - b1*c2;
			const real_t Dt = a1*c2 - c1*a2;

			_center = z0 + Ds / D * n0;
			_radius = (p0 - _center).norm();
		}

	public:
		const vector_t& center() const { return _center; }
		real_t radius() const { return _radius; }

	private:
		//! Center of the circle
		vector_t _center;

		//! Circle radius
		real_t _radius;
	};

	enum class PointCircleClass
	{
		Inside, Outside, OnCircle
	};

	template<typename Scalar, int Dim>
	PointCircleClass isInCircle(const Eigen::Matrix<Scalar, Dim, 1>& p0, const Eigen::Matrix<Scalar, Dim, 1>& p1, const Eigen::Matrix<Scalar, Dim, 1>& p2, const Eigen::Matrix<Scalar, Dim, 1>& p)
	{
		const Eigen::Matrix<Scalar, Dim, 1> p0p = p0 - p;
		const Eigen::Matrix<Scalar, Dim, 1> p1p = p1 - p;
		const Eigen::Matrix<Scalar, Dim, 1> p2p = p2 - p;

		Eigen::Matrix<Scalar, 3, Dim + 1> M;
		M.row(0) << p0p.transpose(), p0p.norm();
		M.row(1) << p1p.transpose(), p1p.norm();
		M.row(2) << p2p.transpose(), p2p.norm();

		const Scalar det = -M.determinant();
		if (det < 0)
			return PointCircleClass::Inside;
		else if (det > 0)
			return PointCircleClass::Outside;
		else
			return PointCircleClass::OnCircle;
	}
}}
