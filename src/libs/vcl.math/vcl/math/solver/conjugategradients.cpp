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
#include <vcl/math/solver/conjugategradients.h>

// C++ standard library
#include <limits>

namespace Vcl { namespace Mathematics { namespace Solver {
	bool ConjugateGradients::solve(ConjugateGradientsContext* ctx, double* residual)
	{
		int dofs = ctx->size();
		if (dofs == 0)
		{
			return false;
		}

		// d = r = b - A*x
		ctx->computeInitialResidual();

		int iteration = 0;
		int sub_iteration = 0;

		while (iteration < dofs && iteration < _maxIterations)
		{
			// i = i + 1
			iteration++;
			sub_iteration++;

			// q = A*d
			ctx->computeQ();

			// d_r = dot(r, r)
			// d_g = dot(d, q)
			// d_b = dot(r, q)
			// d_a = dot(q, q)
			ctx->reduceVectors();

			// alpha = delta_new / (transpose(d) * q)
			// beta = delta_new / delta_old
			// x = x + alpha * d
			// r = r - alpha * q
			// d = r + beta * d
			ctx->updateVectors();

			if (sub_iteration == _chunkSize)
			{
				if (_maxIterations == _chunkSize)
				{
					break;
				}

				// Check if the error is small enough
				double err = ctx->computeError();
				if (err < _eps)
				{
					break;
				}

				// Start a new iteration cycle
				sub_iteration = 0;
			}
		}

		// Finalize the CG
		_iterations = iteration;
		if (residual != nullptr && _maxIterations == _chunkSize)
		{
			ctx->computeError();
		}

		ctx->finish(residual);

		return true;
	}
}}}
