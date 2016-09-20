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
#include <vcl/graphics/trackballcameracontroller.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics
{
	void TrackballCameraController::reset()
	{
		if (!camera())
			return;

		_initialPosition = camera()->position();
		_initialUp       = camera()->up();
		_initialTarget   = camera()->target();
	}

	void TrackballCameraController::setRotationCenter(const Eigen::Vector3f& center)
	{
		if (mode() == CameraMode::Object)
		{
			// Build a rotation transformation around the current center
			auto mC = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{ -_objRotationCenter } };
			auto  R = Eigen::Transform<float, 3, Eigen::Affine>{ _trackball.rotation().toRotationMatrix() };
			auto  C = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{  _objRotationCenter } };

			// Connect to the existing transformation
			auto T = C * R * mC * _objAccumTransform;

			// Store
			_objAccumTransform = T;

			// Set the new center
			_objRotationCenter = center;

			// Reset the trackball
			_trackball.reset();
		}
		else if (mode() == CameraMode::CameraTarget)
		{
			// Build a rotation transformation around the current center
			auto mC = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{ -_objRotationCenter } };
			auto  R = Eigen::Transform<float, 3, Eigen::Affine>{ _trackball.rotation().inverse().toRotationMatrix() };
			auto  C = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{  _objRotationCenter } };

			// Connect to the existing transformation
			auto T = C * R * mC * _objAccumTransform;

			// Store
			_objAccumTransform = T;

			// Set the new center
			_objRotationCenter = center;

			// Reset the trackball
			_trackball.reset();
		}
	}

	void TrackballCameraController::startRotate(float ratio_x, float ratio_y)
	{
		if (!camera())
			return;

		// Reset the the interal state
		reset();

		if (mode() == CameraMode::Object || mode() == CameraMode::CameraTarget)
		{
			if (_trackball.up() != camera()->up())
			{
				// Build a rotation transformation around the current center
				auto mC = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{ -_objRotationCenter } };
				auto  R = Eigen::Transform<float, 3, Eigen::Affine>{ _trackball.rotation().toRotationMatrix() };
				auto  C = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{ _objRotationCenter } };

				// Connect to the existing transformation
				auto T = C * R * mC * _objAccumTransform;

				// Store
				_objCurrTransformation = T.matrix();

				_trackball.reset(camera()->up());
			}

			_trackball.startRotate(ratio_x, ratio_y, true);
		}
		else if (mode() == CameraMode::Camera || mode() == CameraMode::Fly)
		{
			Eigen::Vector3f init_dir{0, 0, 1};
			Eigen::Vector3f curr_dir = (_initialPosition - _initialTarget).normalized();
			Eigen::Quaternionf init_rot = Eigen::Quaternionf::FromTwoVectors(init_dir, curr_dir);

			_trackball.startRotate(Eigen::Quaternionf::Identity(), ratio_x, ratio_y, true);
		}
	}

	void TrackballCameraController::rotate(float ratio_x, float ratio_y)
	{
		using namespace Eigen;

		if (!camera() || !_trackball.isRotating())
			return;

		switch (mode())
		{
		case CameraMode::Object:
		{
			_trackball.rotate(ratio_x, ratio_y, true);

			// Build a rotation transformation around the current center
			auto mC = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{ -_objRotationCenter } };
			auto  R = Eigen::Transform<float, 3, Eigen::Affine>{ _trackball.rotation().toRotationMatrix() };
			auto  C = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{  _objRotationCenter } };

			// Connect to the existing transformation
			auto T = C * R * mC * _objAccumTransform;

			// Store
			_objCurrTransformation = T.matrix();

			break;
		}
		case CameraMode::CameraTarget:
		{
			_trackball.rotate(ratio_x, ratio_y, true);
		
			// Build a rotation transformation around the current center
			auto mC = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{ -_objRotationCenter } };
			auto  R = Eigen::Transform<float, 3, Eigen::Affine>{ _trackball.rotation().inverse().toRotationMatrix() };
			auto  C = Eigen::Transform<float, 3, Eigen::Affine>{ Eigen::Translation3f{  _objRotationCenter } };

			// Connect to the existing transformation
			auto T = C * R * mC * _objAccumTransform;

			// Store
			_objCurrTransformation = T.matrix();

			break;
		}
		case CameraMode::Camera:
		{
			float length = (camera()->position() - camera()->target()).norm();

			_trackball.rotate(ratio_x, ratio_y, true);

			Vector3f point_on_sphere = _initialPosition - _initialTarget;
			point_on_sphere.normalize();
			Vector3f up = _initialUp;

			point_on_sphere = _trackball.rotation().conjugate() * point_on_sphere;
			up = _trackball.rotation().conjugate() * up;
		
			camera()->setPosition(camera()->target() + point_on_sphere * length);
			camera()->setUp(up.normalized());
			break;
		}
		case CameraMode::Fly:
		{
			float length = (camera()->target() - camera()->position()).norm();

			_trackball.rotate(ratio_x, ratio_y, true);
			
			Vector3f point_on_sphere = _initialTarget - _initialPosition;
			point_on_sphere.normalize();
			Vector3f up = point_on_sphere + _initialUp;

			point_on_sphere = _trackball.rotation() * point_on_sphere;
			up = _trackball.rotation() * up;
		
			camera()->setTarget(camera()->position() + point_on_sphere * length);
			camera()->setUp((up - point_on_sphere).normalized());
			break;
		}
		}
	}

	void TrackballCameraController::endRotate()
	{
		if (!camera())
			return;

		_trackball.endRotate();
	}

	void TrackballCameraController::changeZoom(float delta)
	{
		if (!camera())
			return;

		float speed = 0.1f;

		float zoom = camera()->zoom();
		zoom += speed * delta;
		camera()->setZoom(std::max(zoom, 1.0f));
	}

	void TrackballCameraController::move(float x, float y, float z)
	{
		if (!camera())
			return;

		Eigen::Vector3f diff(x, y, z);

		Eigen::Vector3f position = camera()->position() + diff;
		Eigen::Vector3f target = camera()->target() + diff;

		camera()->setPosition(position);
		camera()->setTarget(target);
	}
}}
