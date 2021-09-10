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
#include <vcl/math/solver/conjugategradients.h>
#include <vcl/math/solver/eigenconjugategradientscontext.h>
#include <vcl/math/solver/poisson.h>

namespace Vcl { namespace Mathematics { namespace Solver {
	template<typename Real>
	class Poisson2DCgCtx : public EigenCgBaseContext<Real, Eigen::Dynamic>
	{
		using real_t = Real;
		using vector_t = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;
		using map_t = Eigen::Map<vector_t>;

	public:
		Poisson2DCgCtx(Eigen::Vector2ui dim)
		: EigenCgBaseContext<Real, Eigen::Dynamic>{ dim.x() * dim.y() }
		, _dim{ dim }
		, _rhs{ nullptr, static_cast<Eigen::Index>(dim.x() * dim.y()) }
		{
			for (auto& A : _laplacian)
			{
				A.resize(_dim.x() * _dim.y());
			}
		}

	public:
		void setData(map_t unknowns, map_t rhs)
		{
			this->setX(unknowns);
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
		// d = r = b - A*x
		virtual void computeInitialResidual() override
		{
			auto X = static_cast<const ptrdiff_t>(_dim.x());
			auto Y = static_cast<const ptrdiff_t>(_dim.y());

			const auto& Ac = _laplacian[0];
			const auto& Ax_l = _laplacian[1];
			const auto& Ax_r = _laplacian[2];
			const auto& Ay_l = _laplacian[3];
			const auto& Ay_r = _laplacian[4];

			auto& unknowns = this->_x;
			auto& rhs = _rhs;

			// r = (b - A x)
			//          ---
			//           q
			ptrdiff_t index = 0;
			for (ptrdiff_t sy = 0; sy < Y; sy++)
			{
				for (ptrdiff_t sx = 0; sx < X; sx++, index++)
				{
					float q = 0;
					q += unknowns[index] * Ac[index];
					if (sx > 0)
						q += unknowns[index - 1] * Ax_l[index];
					if (sx < X - 1)
						q += unknowns[index + 1] * Ax_r[index];
					if (sy > 0)
						q += unknowns[index - X] * Ay_l[index];
					if (sy < Y - 1)
						q += unknowns[index + X] * Ay_r[index];

					q = (Ac[index] != 0) ? (rhs[index] - q) : 0;

					this->_res[index] = q;
				}
			}

			this->_dir = this->_res;
		}

		// q = A*d
		virtual void computeQ() override
		{
			const unsigned int X = _dim.x();
			const unsigned int Y = _dim.y();

			const auto& Ac = _laplacian[0];
			const auto& Ax_l = _laplacian[1];
			const auto& Ax_r = _laplacian[2];
			const auto& Ay_l = _laplacian[3];
			const auto& Ay_r = _laplacian[4];

			auto& d = this->_dir;

			ptrdiff_t index = 0;
			for (ptrdiff_t sy = 0; sy < Y; sy++)
			{
				for (ptrdiff_t sx = 0; sx < X; sx++, index++)
				{
					float q = 0;
					q += d[index] * Ac[index];
					if (sx > 0)
						q += d[index - 1] * Ax_l[index];
					if (sx < X - 1)
						q += d[index + 1] * Ax_r[index];
					if (sy > 0)
						q += d[index - X] * Ay_l[index];
					if (sy < Y - 1)
						q += d[index + X] * Ay_r[index];

					q = (Ac[index] != 0) ? q : 0;

					this->_q[index] = q;
				}
			}
		}

	private:
		//! Dimensions of the grid
		Eigen::Vector2ui _dim;

		//! Laplacian matrix (center, x(l/r), y(l/r))
		std::array<vector_t, 5> _laplacian;

		//! Right-hand side
		map_t _rhs;
	};
}}}
