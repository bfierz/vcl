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

// C++ standard library
#include <array>

// VCL configuration
#include <vcl/graphics/cameracontroller.h>
#include <vcl/graphics/trackball.h>

namespace Vcl { namespace Graphics
{
	class TrackballCameraController : public CameraController
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	public:
		const Eigen::Matrix4f currObjectTransformation() const { return _objCurrTransformation; }

	public: // Rotation controls
		void startRotate(float ratio_x, float ratio_y);
		void rotate(float ratio_x, float ratio_y);
		void endRotate();

	public: // Object camera mode
		void setRotationCenter(const Eigen::Vector3f& center);

	public:
		void move(float x, float y, float z);

	public:
		void changeZoom(float delta);

	private:
		void reset();

	private:
		Trackball _trackball;

	private:
		// Initial camera parameters
		Eigen::Vector3f _initialPosition;
		Eigen::Vector3f _initialUp;
		Eigen::Vector3f _initialTarget;

	private: // Paremeters for object camera mode
		//! Current rotation center
		Eigen::Vector3f _objRotationCenter{ Eigen::Vector3f::Zero() };

		//! Current rotation around the object
		Eigen::Matrix4f _objCurrTransformation{ Eigen::Matrix4f::Identity() };

		//! Accumulated rotation around the object
		Eigen::Transform<float, 3, Eigen::Affine> _objAccumTransform{ Eigen::Transform<float, 3, Eigen::Affine>::Identity() };
	};
}}
