/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2016 Basil Fierz
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
#include <vcl/math/solver/jacobi.h>
#include <vcl/math/solver/poisson.h>

namespace Vcl { namespace Mathematics { namespace Solver {
	template<typename Real>
	class Poisson2DJacobiCtx : public JacobiContext
	{
		using real_t = Real;
		using vector_t = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;
		using map_t = Eigen::Map<vector_t>;

	public:
		Poisson2DJacobiCtx(Eigen::Vector2ui dim)
		: _dim(dim)
		, _unknowns(nullptr, dim.x() * dim.y())
		, _rhs(nullptr, dim.x() * dim.y())
		{
			_next.setZero(dim.x() * dim.y());
			for (auto& A : _laplacian)
			{
				A.resize(_dim.x() * _dim.y());
			}
		}

	public:
		void setData(map_t unknowns, map_t rhs)
		{
			new (&_unknowns) map_t(unknowns);
			new (&_rhs) map_t(rhs);
		}

		void updatePoissonStencil(real_t h, real_t k, real_t o, Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip)
		{
			auto& Ac = _laplacian[0];
			auto& Ax_l = _laplacian[1];
			auto& Ax_r = _laplacian[2];
			auto& Ay_l = _laplacian[3];
			auto& Ay_r = _laplacian[4];

			makePoissonStencil(
				_dim, h, k, o, map_t{ Ac.data(), Ac.size() },
				map_t{ Ax_l.data(), Ax_l.size() }, map_t{ Ax_r.data(), Ax_r.size() },
				map_t{ Ay_l.data(), Ay_l.size() }, map_t{ Ay_r.data(), Ay_r.size() },
				skip);
		}

	public:
		virtual int size() const override
		{
			return _dim.x() * _dim.y();
		}

	public:
		//
		virtual void precompute() override
		{
			_error = 0;
		}

		// A x = b
		// -> A = D + R
		// -> x^{n+1} = D^-1 (b - R x^{n})
		virtual void updateSolution() override
		{
			auto X = static_cast<const ptrdiff_t>(_dim.x());
			auto Y = static_cast<const ptrdiff_t>(_dim.y());

			const auto& Ac = _laplacian[0];
			const auto& Ax_l = _laplacian[1];
			const auto& Ax_r = _laplacian[2];
			const auto& Ay_l = _laplacian[3];
			const auto& Ay_r = _laplacian[4];

			auto& unknowns = _unknowns;
			auto& rhs = _rhs;

			// Error
			float acc = 0;

			// x^{n+1} = D^-1 (b - R x^{n})
			//                -------------
			//                      q
			ptrdiff_t index = 0;
			for (ptrdiff_t sy = 0; sy < Y; sy++)
			{
				for (ptrdiff_t sx = 0; sx < X; sx++, index++)
				{
					float q = 0;
					if (sx > 0)
						q += unknowns[index - 1] * Ax_l[index];
					if (sx < X - 1)
						q += unknowns[index + 1] * Ax_r[index];
					if (sy > 0)
						q += unknowns[index - X] * Ay_l[index];
					if (sy < Y - 1)
						q += unknowns[index + X] * Ay_r[index];

					float n = (rhs[index] - q) / Ac[index];
					n = (Ac[index] != 0) ? n : unknowns[index];

					_next[index] = n;

					// Compute the error
					if (Ac[index])
					{
						float e = rhs[index] - (Ac[index] * unknowns[index] + q);
						acc += e * e;
					}
				}
			}

			unknowns = _next;
			_error = acc;
		}

		//
		virtual double computeError() override
		{
			return sqrt(_error) / size();
		}

		//! Ends the solver and returns the residual
		virtual void finish(double*) override {}

	private:
		//! Dimensions of the grid
		Eigen::Vector2ui _dim;

		//! Current error
		float _error{ 0 };

		//! Laplacian matrix (center, x(l/r), y(l/r))
		std::array<vector_t, 5> _laplacian;

		//! Left-hand side
		map_t _unknowns;

		//! Right-hand side
		map_t _rhs;

		//! Temporary buffer for the updated solution
		vector_t _next;
	};
}}}
