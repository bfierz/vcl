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
#include <vcl/config/eigen.h>

// C++ standard library
#include <array>

// VCL
#include <vcl/physics/fluid/centergrid.h>

namespace Vcl { namespace Physics { namespace Fluid
{
	/*!
	 *	\brief Simple grid based fluid simulation
	 */
	class EulerFluidSimulation
	{
	public:
		EulerFluidSimulation();
		virtual ~EulerFluidSimulation();

	public:
		virtual void update(Fluid::CenterGrid& g, double dt);

	protected:
		void addBuoyancy(float* forceZ, float buoyancy, const float* source, const Eigen::Vector3i& dim);

		void setBorderZero (float* field, const Eigen::Vector3i& dim);
		void setBorderZeroX(float* field, const Eigen::Vector3i& dim);
		void setBorderZeroY(float* field, const Eigen::Vector3i& dim);
		void setBorderZeroZ(float* field, const Eigen::Vector3i& dim);

		void setNeumannX(float* field, const Eigen::Vector3i& dim);
		void setNeumannY(float* field, const Eigen::Vector3i& dim);
		void setNeumannZ(float* field, const Eigen::Vector3i& dim);

		void copyBorderX(float* field, const Eigen::Vector3i& dim);
		void copyBorderY(float* field, const Eigen::Vector3i& dim);
		void copyBorderZ(float* field, const Eigen::Vector3i& dim);

	private:
	};
}}}
