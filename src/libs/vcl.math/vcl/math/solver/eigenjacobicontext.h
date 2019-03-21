/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <vcl/math/solver/jacobi.h>

namespace Vcl { namespace Mathematics { namespace Solver
{
	template <typename MatrixT>
	class EigenJacobiContext : public JacobiContext
	{
	public:
		using real_t   = typename MatrixT::Scalar;
		using vector_t = Eigen::Matrix<real_t, MatrixT::ColsAtCompileTime, 1>;
		using matrix_t = Eigen::Matrix<real_t, MatrixT::RowsAtCompileTime, MatrixT::ColsAtCompileTime>;
		using map_t    = Eigen::Map<vector_t>;

	public:
		EigenJacobiContext(const matrix_t* A, const vector_t* b)
		: _A(A)
		, _x(nullptr, b->size())
		, _b(b)
		, _size(b->size())
		{
			_D = A->diagonal();
			_DInv = A->diagonal().cwiseInverse();
			_R = *A;
			_R.diagonal().setZero();
		}

		void setX(map_t x)
		{
			new(&_x) map_t(x);
		}

		//! \brief Implement virtual interface
		//! \{
		int size() const override
		{
			return static_cast<int>(_size);
		}
		void precompute() override
		{
			_error = 0;
		}
		void updateSolution() override
		{
			// x^{n+1} = D^-1 (b - R x^{n})
			//                -------------
			//                      q
			auto& x = _x;
			auto& b = *_b;
			_next = _DInv.asDiagonal() * (b - _R * x);

			x = _next;
		}
		 double computeError() override
		{
			auto& A = *_A;
			auto& x = _x;
			auto& b = *_b;

			_error = (b - A * x).norm();

			return _error / size();
		}
		void finish(double * residual) override
		{
			if (residual)
				(*residual) = _error;
		}
		//! \}

	protected:
		const matrix_t* _A;
		map_t _x;
		const vector_t* _b;

	private:
		//! Current error
		real_t _error{ 0 };
		
		//! Size of the problem
		size_t _size;

		//! System matrix diagonal
		vector_t _D;

		//! Per-element inverse of the system matrix diagonal
		vector_t _DInv;
		
		//! Off-diagonal elements of the system matrix
		matrix_t _R;

		//! Temporary buffer for the updated solution
		vector_t _next;
	};
}}}
