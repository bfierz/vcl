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
#include <vcl/math/math.h>

namespace Vcl { namespace Geometry
{
	/*!
	 *	Implementation based on the summary in
	 *	http://tavianator.com/fast-branchless-raybounding-box-intersections/
	 *	http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
	 */
	bool intersects
	(
		const Eigen::AlignedBox<float, 3>& box,
		const Eigen::ParametrizedLine<float, 3>& ray
	)
	{
		using namespace Vcl::Mathematics;

		float tmin = -std::numeric_limits<float>::infinity();
		float tmax =  std::numeric_limits<float>::infinity();

		for (int i = 0; i < 3; ++i)
		{
			float t1 = (box.min()[i] - ray.origin()[i]) / ray.direction()[i];
			float t2 = (box.max()[i] - ray.origin()[i]) / ray.direction()[i];

			tmin = max(tmin, min(min(t1, t2), tmax));
			tmax = min(tmax, max(max(t1, t2), tmin));
		}

		return tmax > max(tmin, 0.0f);
	}
	template<typename Real, int Width>
	Vcl::VectorScalar<bool, Width> intersects
	(
		const Eigen::AlignedBox<Vcl::VectorScalar<Real, Width>, 3>& box,
		const Eigen::ParametrizedLine<Vcl::VectorScalar<Real, Width>, 3>& ray
	)
	{
		using namespace Vcl::Mathematics;

		Vcl::VectorScalar<Real, Width> tmin = -std::numeric_limits<float>::infinity();
		Vcl::VectorScalar<Real, Width> tmax =  std::numeric_limits<float>::infinity();

		for (int i = 0; i < 3; ++i)
		{
			Vcl::VectorScalar<Real, Width> t1 = (box.min()[i] - ray.origin()[i]) / ray.direction()[i];
			Vcl::VectorScalar<Real, Width> t2 = (box.max()[i] - ray.origin()[i]) / ray.direction()[i];

			tmin = max(tmin, min(min(t1, t2), tmax));
			tmax = min(tmax, max(max(t1, t2), tmin));
		}

		return tmax > max(tmin, 0.0f);
	}
}}
