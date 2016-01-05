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
	class RotationAroundCenter
	{
	public:
		//RotationAroundCenter(const Eigen::Quaternionf& R, const Eigen::Vector3f& c) : _rotation(R), _center(c) {}
		RotationAroundCenter(const Eigen::Quaternionf& R, const Eigen::Vector3f& c)
		{
			auto A = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{ -c } };
			auto B = Eigen::Transform<float, 3, Eigen::Affine>{ R.toRotationMatrix() };
			auto C = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{  c } };

			_transform = C * B * A;
		}
		RotationAroundCenter(const Eigen::Transform<float, 3, Eigen::Affine>& T)
		: _transform(T)
		{

		}

		const Eigen::Matrix4f toMatrix() const { return _transform.matrix(); }

		//const Eigen::Quaternionf& rotation() const { return _rotation; }
		//const Eigen::Vector3f&    center()   const { return _center; }

	public:
		RotationAroundCenter operator* (const RotationAroundCenter& rhs) const
		{
			return _transform * rhs._transform;
		}
		//RotationAroundCenter operator* (const RotationAroundCenter& rhs) const
		//{
		//	Eigen::Quaternionf combined_rot = (_rotation*rhs._rotation).normalized();
		//
		//	return{ combined_rot, (combined_rot * -rhs._center) + (_rotation * rhs._center) + (_rotation * -_center) + _center };
		//}

	private:
		//Eigen::Quaternionf _rotation{ Eigen::Quaternionf::Identity() };
		//Eigen::Vector3f    _center  { Eigen::Vector3f::Zero() };
		Eigen::Transform<float, 3, Eigen::Affine> _transform;
	};

	class TrackballCameraController : public CameraController
	{
	public:
		const Eigen::Matrix4f currObjectTransformation() const { return _objCurrTransformation; }

	public: // Rotation controls
		void setRotationCenter(const Eigen::Vector3f& center);
		void startRotate(float ratio_x, float ratio_y);
		void rotate(float ratio_x, float ratio_y);
		void endRotate();

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
		Eigen::Vector3f    _objRotationCenter{ Eigen::Vector3f::Zero() };

		RotationAroundCenter _objAccumTransform{ Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero() };

		Eigen::Matrix4f _objCurrTransformation{ Eigen::Matrix4f::Identity() };
	};
}}
