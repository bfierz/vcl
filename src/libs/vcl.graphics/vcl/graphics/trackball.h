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

namespace Vcl { namespace Graphics
{
	class Trackball
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	public:
		void reset(const Eigen::Vector3f& up = { 0.0f, 1.0f, 0.0f });
		void startRotate(float ratio_x, float ratio_y, bool right_handed = true);
		void startRotate(const Eigen::Quaternionf& inital_rotation, float ratio_x, float ratio_y, bool right_handed = true);
		void rotate(float ratio_x, float ratio_y, bool right_handed = true);
		void endRotate();

	public:
		bool isRotating() const { return _rotate; }
		const Eigen::Vector3f& up() const { return _up; }
		const Eigen::Quaternionf& rotation() const { return _lastQuat; }

	private:
		Eigen::Vector3f project(float ratio_x, float ratio_y, bool right_handed);
		Eigen::Quaternionf fromPosition(Eigen::Vector3f v) const;

	private:
		//! Up vector
		Eigen::Vector3f _up{ 0.0f, 1.0f, 0.0f };

		//! Radius of the trackball
		float _radius{ 1.0f };

		//! Last rotation
		Eigen::Vector3f _lastPosition{ Eigen::Vector3f::Zero() };

		//! Current rotation
		Eigen::Quaternionf _lastQuat{ Eigen::Quaternionf::Identity() };

		//! Are we tracking the rotation
		bool _rotate{ false };
	};
}}
