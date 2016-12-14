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

// GSL
#include <gsl>

// VCL
#include <vcl/math/solver/jacobi.h>
#include <vcl/math/solver/poisson.h>

namespace Vcl { namespace Mathematics { namespace Solver
{
	template<typename Real>
	class Poisson3DJacobiCtx : public JacobiContext
	{
		using real_t = Real;
		using vector_t = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;
		using map_t = Eigen::Map<vector_t>;

	public:
		Poisson3DJacobiCtx(Eigen::Vector3ui dim)
		: _dim(dim)
		{
			_next.setZero(dim.x()*dim.y()*dim.z());
		}
		
	public:
		void setData(gsl::not_null<map_t*> unknowns, gsl::not_null<map_t*> rhs)
		{
			_unknowns = unknowns;
			_rhs = rhs;
		}

		void updatePoissonStencil(real_t h, real_t k, Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip)
		{
			auto& Ac = _laplacian[0];
			auto& Ax_l = _laplacian[1];
			auto& Ax_r = _laplacian[2];
			auto& Ay_l = _laplacian[3];
			auto& Ay_r = _laplacian[4];
			auto& Az_l = _laplacian[5];
			auto& Az_r = _laplacian[6];

			Ac.resize(_dim.x() * _dim.y() * _dim.z());
			Ax_l.resize(_dim.x() * _dim.y() * _dim.z());
			Ax_r.resize(_dim.x() * _dim.y() * _dim.z());
			Ay_l.resize(_dim.x() * _dim.y() * _dim.z());
			Ay_r.resize(_dim.x() * _dim.y() * _dim.z());
			Az_l.resize(_dim.x() * _dim.y() * _dim.z());
			Az_r.resize(_dim.x() * _dim.y() * _dim.z());

			makePoissonStencil
			(
				_dim, h, k, map_t{ Ac.data(), Ac.size() },
				map_t{ Ax_l.data(), Ax_l.size() }, map_t{ Ax_r.data(), Ax_r.size() },
				map_t{ Ay_l.data(), Ay_l.size() }, map_t{ Ay_r.data(), Ay_r.size() },
				map_t{ Az_l.data(), Az_l.size() }, map_t{ Az_r.data(), Az_r.size() },
				skip
			);
		}

	public:
		virtual int size() const override
		{
			return _dim.x()*_dim.y()*_dim.z();
		}

	public:
		//
		virtual void precompute() override
		{
			*_unknowns = *_rhs;
			_error = 0;
		}

		// A x = b
		// -> A = D + R
		// -> x^{n+1} = D^-1 (b - R x^{n})
		virtual void updateSolution() override
		{
			const unsigned int X = _dim.x();
			const unsigned int Y = _dim.y();
			const unsigned int Z = _dim.z();

			const auto& Ac = _laplacian[0];
			const auto& Ax_l = _laplacian[1];
			const auto& Ax_r = _laplacian[2];
			const auto& Ay_l = _laplacian[3];
			const auto& Ay_r = _laplacian[4];
			const auto& Az_l = _laplacian[5];
			const auto& Az_r = _laplacian[6];

			auto& unknowns = *_unknowns;
			auto& rhs = *_rhs;

			// Error
			float acc = 0;

			// x^{n+1} = D^-1 (b - R x^{n})
			//                -------------
			//                      q
			size_t index = X*Y + X + 1;
			for (size_t sz = 1; sz < Z - 1; sz++, index += 2 * X)
			{
				for (size_t sy = 1; sy < Y - 1; sy++, index += 2)
				{
					for (size_t sx = 1; sx < X - 1; sx++, index++)
					{
						float q =
							unknowns[index - 1  ] * Ax_l[index] +
							unknowns[index + 1  ] * Ax_r[index] +
							unknowns[index - X  ] * Ay_l[index] +
							unknowns[index + X  ] * Ay_r[index] +
							unknowns[index - X*Y] * Az_l[index] +
							unknowns[index + X*Y] * Az_r[index];

						float n = (rhs[index] - q) / Ac[index];
						n = (Ac[index] != 0) ? n : 0;

						_next[index] = n;

						// Compute the error
						if (Ac[index])
						{
							float e = rhs[index] - (Ac[index] * unknowns[index] + q);
							acc += e*e;
						}
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
		virtual void finish(double* residual) override {}

	private:
		//! Dimensions of the grid
		Eigen::Vector3ui _dim;

		//! Current error
		float _error{ 0 };

		//! Laplacian matrix (center, x(l/r), y(l/r), z(l/r))
		std::array<vector_t, 7> _laplacian;
		
		//! Left-hand side 
		map_t* _unknowns;

		//! Right-hand side
		map_t* _rhs;

		//! Temporary buffer for the updated solution
		vector_t _next;
	};
}}}
