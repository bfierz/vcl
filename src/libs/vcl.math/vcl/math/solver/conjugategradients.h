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

// C++ standard library
#include <limits>

namespace Vcl { namespace Mathematics { namespace Solver {
	class ConjugateGradientsContext
	{
	public:
		virtual ~ConjugateGradientsContext() = default;

		//! Size of the problem to be solved
		virtual int size() const = 0;

	public:
		// d = r = b - A*x
		virtual void computeInitialResidual() = 0;

		// q = A*d
		virtual void computeQ() = 0;

		// d_r = dot(r, r)
		// d_g = dot(d, q)
		// d_b = dot(r, q)
		// d_a = dot(q, q)
		virtual void reduceVectors() = 0;

		// alpha = d_r / d_g;
		// beta = (d_r - 2.0f * alpha * d_b + alpha * alpha * d_a) / d_r;
		// x = x + alpha * d
		// r = r - alpha * q
		// d = r + beta * d
		virtual void updateVectors() = 0;

		// abs(beta * d_r);
		virtual double computeError() = 0;

		//! Ends the solver and returns the residual
		virtual void finish(double* residual) = 0;
	};

	class ConjugateGradients
	{
	public:
		virtual ~ConjugateGradients() = default;

		void setPrecision(double eps) { _eps = eps; }
		void setMaxIterations(int iter) { _maxIterations = iter; }
		void setIterationChunkSize(int size) { _chunkSize = size; }

	public:
		int nrIterations() const { return _iterations; }

	public:
		virtual bool solve(ConjugateGradientsContext* ctx, double* residual = nullptr);

	private: // Solver configuration
		//! Maximum number of iterations
		int _maxIterations = 0;
		int _chunkSize = 1;
		double _eps = std::numeric_limits<double>::epsilon();

	private: // Meta results
		//! Number of iterations the solver needed
		int _iterations = 0;
	};
}}}
