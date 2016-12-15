/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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
#include <vcl/math/solver/conjugategradients.h>

namespace Vcl { namespace Mathematics { namespace Solver
{
	template <typename Real, int ProblemSize = Eigen::Dynamic>
	class EigenCgBaseContext : public ConjugateGradientsContext
	{
	public:
		using real_t  = Real;
		using vector_t = Eigen::Matrix<real_t, ProblemSize, 1>;
		using map_t = Eigen::Map<vector_t>;

	public:
		EigenCgBaseContext(size_t s)
		: _size(s)
		{
			if (s > 0)
			{
				_dir = vector_t::Zero(_size);
				_q = vector_t::Zero(_size);
				_res = vector_t::Zero(_size);
			}
		}

	public:
		virtual int size() const override
		{
			return static_cast<int>(_size);
		}

		void setX(map_t* x)
		{
			_x = x;
		}

	public:
		// d = r = b - A*x
		virtual void computeInitialResidual() =0;
			
		// q = A*d
		virtual void computeQ() =0;
			
		// d_r = dot(r, r)
		// d_g = dot(d, q)
		// d_b = dot(r, q)
		// d_a = dot(q, q)
		void reduceVectors()
		{
			real_t d_r = _res.squaredNorm();
			real_t d_g = _dir.dot(_q);
			real_t d_b = _res.dot(_q);
			real_t d_a = _q.squaredNorm();
				
			_alpha = 0.0f;
			if (fabs(d_g) > 0.0f)
				_alpha = d_r / d_g;

			_beta = d_r - 2.0f * _alpha * d_b + _alpha * _alpha * d_a;
			if (fabs(d_r) > 0.0f)
				_beta = _beta / d_r;

			_residualLength = d_r;
		}

		// alpha = d_r / d_g;
		// beta = d_r - 2.0f * alpha * d_b + alpha * alpha * d_a;
		// x = x + alpha * d
		// r = r - alpha * q
		// d = r + beta * d
		void updateVectors()
		{
			Require(_x != nullptr, "Solution vector is set.");

			(*_x) += _alpha * _dir;
			_res -= _alpha * _q;
			_dir = _res + _beta * _dir;
		}

		// abs(beta * d_r);
		virtual double computeError() override
		{
			return abs(_beta * _residualLength);
		}
			
		virtual void finish(double* residual = nullptr) override
		{
			if (residual)
				(*residual) = _residualLength;
		}

	protected: // Matrix to solve
		map_t* _x{ nullptr };
		size_t _size;

	private:
		real_t _alpha;
		real_t _beta;
		real_t _residualLength;

	protected: // Temporary buffers
		vector_t _dir;
		vector_t _q;
		vector_t _res;
	};

	template <typename MatrixT>
	class GenericEigenCgContext : public EigenCgBaseContext<typename MatrixT::Scalar, MatrixT::RowsAtCompileTime>
	{
	public:
		using matrix_t = MatrixT;
		using vector_t = typename EigenCgBaseContext<typename MatrixT::Scalar, MatrixT::RowsAtCompileTime>::vector_t;

	public:
		GenericEigenCgContext(const matrix_t* A, const vector_t* b, vector_t* x)
		: EigenCgBaseContext(x)
		, _M(A)
		, _b(b)
		{
		}

	public:
		// d = r = b - A*x
		virtual void computeInitialResidual() override
		{
			_res = (*_b) - (*_M) * (*_x);
			_dir = _res;
		}
			
		// q = A*d
		virtual void computeQ() override
		{
			_q = (*_M) * _dir;
		}

	private: /* Matrix to solve */
		const matrix_t* _M;
		const vector_t* _b;
	};
}}}
