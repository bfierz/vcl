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

// Eigen
#include <Eigen/Geometry>

namespace Vcl { namespace Geometry
{
	template<typename Scalar, int Dim>
	class Ray
	{
	public:
		using int_t = typename VectorTypes<Scalar>::int_t;
		using vector_t = typename Eigen::ParametrizedLine<Scalar, Dim>::VectorType;

	public:
		Ray(const vector_t& o, const vector_t& d)
		: _ray(o, d)
		{
			_invDir = _ray.direction().cwiseInverse();
			for (int i = 0; i < Dim; i++)
				_signs[i] = select(_invDir[i] < 0, int_t{ 1 }, int_t{ 0 });
		}

	public:
		operator const Eigen::ParametrizedLine<Scalar, Dim>() const
		{
			return _ray;
		}

		const vector_t& origin() const { return _ray.origin(); }
		const vector_t& direction() const { return _ray.direction(); }
		const vector_t& invDirection() const { return _invDir; }
		const Eigen::Matrix<int_t, Dim, 1>& signs() const { return _signs; }

	public:
		vector_t operator() (const Scalar& t) const
		{
			return _ray.pointAt(t);
		}

	private:
		//! Encapsulated ray object
		Eigen::ParametrizedLine<Scalar, Dim> _ray;

		//! Inverted ray direction
		vector_t _invDir;

		//! Signs
		Eigen::Matrix<int_t, Dim, 1> _signs;
	};
}}
