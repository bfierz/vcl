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

// C++ standard library
#include <array>

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/contract.h>
#include <vcl/geometry/ray.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Geometry
{
	/*!
	 *	\brief Ray-AABB intersection
	 *
	 *	Implementation based on the summary in
	 *	http://tavianator.com/fast-branchless-raybounding-box-intersections/
	 *	http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
	 *
	 *	\note Rays aligned with the border of the bounding box produce only very inconsistent intersections
	 */
	bool intersects_Barnes
	(
		const Eigen::AlignedBox<float, 3>& box,
		const Ray<float, 3>& ray
	)
	{
		using namespace Vcl::Mathematics;

		float tmin = -std::numeric_limits<float>::infinity();
		float tmax =  std::numeric_limits<float>::infinity();

		for (int i = 0; i < 3; ++i)
		{
			float t1 = (box.min()[i] - ray.origin()[i]) * ray.invDirection()[i];
			float t2 = (box.max()[i] - ray.origin()[i]) * ray.invDirection()[i];

			tmin = max(tmin, min(min(t1, t2),  std::numeric_limits<float>::infinity()));
			tmax = min(tmax, max(max(t1, t2), -std::numeric_limits<float>::infinity()));
		}

		return tmax > max(tmin, { 0.0f });
	}

	/*!
	 *	\brief Ray-AABB intersection
	 *
	 *	Implementation from
	 *	https://www.solidangle.com/research/jcgt2013_robust_BVH-revised.pdf
	 */
	bool intersects_MaxMult
	(
		const Eigen::AlignedBox<float, 3>& box,
		const Ray<float, 3>& r
	)
	{
		using namespace Vcl::Mathematics;

		float txmin, txmax, tymin, tymax, tzmin, tzmax;

		Eigen::Vector3f bounds[] = { box.min(), box.max() };

		txmin = (bounds[    r.signs().x()].x() - r.origin().x()) * r.invDirection().x();
		txmax = (bounds[1 - r.signs().x()].x() - r.origin().x()) * r.invDirection().x();
		tymin = (bounds[    r.signs().y()].y() - r.origin().y()) * r.invDirection().y();
		tymax = (bounds[1 - r.signs().y()].y() - r.origin().y()) * r.invDirection().y();
		tzmin = (bounds[    r.signs().z()].z() - r.origin().z()) * r.invDirection().z();
		tzmax = (bounds[1 - r.signs().z()].z() - r.origin().z()) * r.invDirection().z();

		// Disallow any intersection that lies behind the start point of the ray
		float tmin = 0;
		float tmax = std::numeric_limits<float>::infinity();

		tmin = max(tzmin, max(tymin, max(txmin, tmin)));
		tmax = min(tzmax, min(tymax, min(txmax, tmax)));
		tmax *= 1.00000024f;

		return tmin <= tmax;
	}
	
	template<typename Real, int Width>
	Vcl::VectorScalar<bool, Width> intersects_MaxMult
	(
		const Eigen::AlignedBox<Vcl::VectorScalar<Real, Width>, 3>& box,
		const Ray<Vcl::VectorScalar<Real, Width>, 3>& r
	)
	{
		using namespace Vcl::Mathematics;

		using real_t = Vcl::VectorScalar<Real, Width>;

		real_t txmin, txmax, tymin, tymax, tzmin, tzmax;

		txmin = (select(r.signs().x() == 0, box.min().x(), box.max().x()) - r.origin().x()) * r.invDirection().x();
		txmax = (select(r.signs().x() == 1, box.min().x(), box.max().x()) - r.origin().x()) * r.invDirection().x();
		tymin = (select(r.signs().y() == 0, box.min().y(), box.max().y()) - r.origin().y()) * r.invDirection().y();
		tymax = (select(r.signs().y() == 1, box.min().y(), box.max().y()) - r.origin().y()) * r.invDirection().y();
		tzmin = (select(r.signs().z() == 0, box.min().z(), box.max().z()) - r.origin().z()) * r.invDirection().z();
		tzmax = (select(r.signs().z() == 1, box.min().z(), box.max().z()) - r.origin().z()) * r.invDirection().z();

		// Disallow any intersection that lies behind the start point of the ray
		real_t tmin = 0;
		real_t tmax = std::numeric_limits<float>::infinity();

		tmin = max(tzmin, max(tymin, max(txmin, tmin)));
		tmax = min(tzmax, min(tymax, min(txmax, tmax)));
		tmax *= 1.00000024f;

		return tmin <= tmax;
	}

	/*!
	*	\brief Ray-AABB intersection
	*
	*	Method from Pharr, Humphrey
	*/
	bool intersects_Pharr
	(
		const Eigen::AlignedBox<float, 3>& box,
		const Ray<float, 3>& ray
	)
	{
		using namespace Vcl::Mathematics;

		float t0 = 0;
		float t1 = std::numeric_limits<float>::infinity();

		for (int i = 0; i < 3; ++i)
		{
			float tNear = (box.min()[i] - ray.origin()[i]) * ray.invDirection()[i];
			float tFar  = (box.max()[i] - ray.origin()[i]) * ray.invDirection()[i];

			if (tNear > tFar) std::swap(tNear, tFar);
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar < t1  ? tFar : t1;

			if (t0 > t1) return false;
		}

		return true;
	}
}}
