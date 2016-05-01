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

// VCL
#include <vcl/core/contract.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Geometry
{
	template<typename Scalar, int Dim>
	class Tetrahedron
	{
	public:
		using real_t = Scalar;
		using vector_t = Eigen::Matrix<Scalar, Dim, 1>;

	public:
		Tetrahedron(const vector_t& a, const vector_t& b, const vector_t& c, const vector_t& d)
		{
			_data[0] = a;
			_data[1] = b;
			_data[2] = c;
			_data[3] = d;
		}

	public:
		const vector_t& operator[] (size_t idx) const
		{
			Require(idx < 4, "Id is in [0, 4[");

			return _data[idx];
		}

	public:
		vector_t computeCenter() const
		{
			return (real_t) 0.25 * (_data[0] + _data[1] + _data[2] + _data[3]);
		}

		real_t inradius() const
		{
			vector_t L[6];
			L[0] = _data[1] - _data[0];
			L[1] = _data[2] - _data[1];
			L[2] = _data[0] - _data[2];
			L[3] = _data[3] - _data[0];
			L[4] = _data[3] - _data[1];
			L[5] = _data[3] - _data[2];

			ScalarT A = (real_t) 0.5 * (L[2].cross(L[0]).norm() +
			                            L[3].cross(L[0]).norm() +
			                            L[4].cross(L[1]).norm() +
			                            L[3].cross(L[2]).norm());

			return (real_t) 3 * computeVolume() / A;
		}

		real_t computeCircumradius() const
		{
			vector_t L[6];
			L[0] = _data[1] - _data[0];
			L[1] = _data[2] - _data[1];
			L[2] = _data[0] - _data[2];
			L[3] = _data[3] - _data[0];
			L[4] = _data[3] - _data[1];
			L[5] = _data[3] - _data[2];

			real_t result = (L[3].squaredNorm() * L[2].cross(L[0]) + L[2].squaredNorm() * L[3].cross(L[0]) + L[0].squaredNorm() * L[3].cross(L[2])).norm();
			return real_t(1) / real_t(12) * result / computeVolume();
		}

		real_t computeVolume() const
		{
			real_t vol = computeSignedVolume();

			Ensure(vol >= 0, "Volume is positive.");
			return abs(vol);
		}

		real_t computeSignedVolume() const
		{
			using Vcl::Mathematics::equal;

			// Calculate the volume for a right-handed coordinate system
			Eigen::Matrix<real_t, 4, 4> m;
			m << _data[0].x(), _data[1].x(), _data[2].x(), _data[3].x(),
				 _data[0].y(), _data[1].y(), _data[2].y(), _data[3].y(),
				 _data[0].z(), _data[1].z(), _data[2].z(), _data[3].z(),
				   (real_t) 1,   (real_t) 1,   (real_t) 1,   (real_t) 1;

			real_t vol = -m.determinant() / (real_t) 6;
		
			AssertBlock
			{
				real_t ref = (_data[3]- _data[0]).dot((_data[1]- _data[0]).cross((_data[2]- _data[0]))) / (real_t) 6;

				Ensure(equal(vol, ref, (real_t) 1e-6), "Volumes are equal.");
			}

			return vol;
		}

		real_t computeHeight(unsigned int i) const
		{
			Require(i < 4, "Id is in [0, 4[");

			vector_t pa = _data[0];
			vector_t pb = _data[1];
			vector_t pc = _data[2];
			vector_t pd = _data[3];

			switch (i)
			{
			case 0:
			{
				return (pd - pb).cross(pc - pb).normalized().dot(pa - pb);
			}
			case 1:
			{
				return (pc - pa).cross(pd - pa).normalized().dot(pb - pa);
			}
			case 2:
			{
				return (pd - pa).cross(pb - pa).normalized().dot(pc - pa);
			}
			case 3:
			{
				return (pb - pa).cross(pc - pa).normalized().dot(pd - pa);
			}
			}

			return 0;
		}

		real_t computeAngle(const vector_t& a, const vector_t& b) const
		{
			real_t l1 = a.squaredNorm();
			real_t l2 = b.squaredNorm();
			real_t tmp = (a.dot(b)) / sqrt(l1*l2);
			if (tmp >= 1.0)
				return 0.0f;						// avoid rounding errors
			if (tmp <= -1.0)
				return static_cast<real_t>(M_PI);	// avoid rounding errors

			tmp = (ScalarT)acos(tmp);

			return tmp;
		}

	private:
		vector_t _data[4];
	};
}}
