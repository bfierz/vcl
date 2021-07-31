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
#include <vcl/math/solver/jacobi.h>

namespace Vcl { namespace Mathematics { namespace Solver
{
	bool Jacobi::solve(JacobiContext* ctx, double* residual)
	{
		int dofs = ctx->size();
		if (dofs == 0)
		{
			return false;
		}

		// A x = b
		// -> A = D + R
		// -> x^{n+1} = D^-1 (b - R x^{n})

		// -> c = D^-1 b
		// -> C = D^-1 R
		//      = D^-1 (A - D)
		//      = D^-1 A - I
		//   -C = I - D^-1 A

		// -> x^{n+1} = D^-1 b + (I - D^-1 A) x^{n}
		// -> x^{n+1} = c + C x^{n}

		//  c = D^-1 b
		// -C = I - D^-1 A
		ctx->precompute();

		int iteration = 0;
		int sub_iteration = 0;

		while (iteration < _maxIterations)
		{
			// i = i + 1
			iteration++;
			sub_iteration++;

			// x^{n+1} = c + C x^{n}
			ctx->updateSolution();

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
