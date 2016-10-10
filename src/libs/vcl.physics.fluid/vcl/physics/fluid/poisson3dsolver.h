/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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

// VCL

// VCL forward declarations
namespace Vcl { namespace Physics { namespace Fluid { class CenterGrid; }}}

namespace Vcl { namespace Physics { namespace Fluid
{
	class CenterGrid3DPoissonSolver
	{
	public:
		int nrInterations() const { return _iterations; }
		float residualLength() const { return _residualLength; }

	public:
		virtual void updateSolver(Fluid::CenterGrid& g) = 0;
		virtual void makeDivergenceFree(Fluid::CenterGrid& g) = 0;
		virtual void diffuseField(Fluid::CenterGrid& g, float diffusion_constant) = 0;

	private: // Solver configuration

		//! Maximum number of executed solver iterations (No limit: 0)
		int _maxIterations = 0;

		//! Desired accuracy of the solution
		float _eps = 1e-6f;

	private: // Results
		//! Number of iterations the last solver execution took
		int _iterations = 0;

		//! Residual of the last solver run
		float _residualLength = 0.0f;
	};
}}}
