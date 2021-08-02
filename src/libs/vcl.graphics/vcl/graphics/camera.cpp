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

// Header
#include <vcl/graphics/camera.h>

// C++ standard libary
#include <cmath>

#ifndef M_PI
#	define M_PI 3.1415926535897932384626433832795
#endif

namespace Vcl { namespace Graphics {
	Camera::Camera(std::shared_ptr<MatrixFactory> factory)
	: Camera({ 1, 1, 1 }, { 0, 0, 0 }, std::move(factory))
	{
	}

	Camera::Camera(Eigen::Vector3f position, Eigen::Vector3f target, std::shared_ptr<MatrixFactory> factory)
	: _factory(std::move(factory))
	, _position(position)
	, _target(target)
	, _useTarget(true)
	, _direction(0, 0, 0)
	, _useDirection(false)
	, _up(0, 1, 0)
	, _fov((float)M_PI / 4.0f)
	, _nearPlane(0.01f)
	, _farPlane(100.0f)
	, _zoom(1.0f)
	, _viewportX(512)
	, _viewportY(512)
	{
		_changedView = true;
		_changedProjection = true;
	}

	void Camera::setPosition(const Eigen::Vector3f& position)
	{
		_changedView = true;
		_position = position;
	}

	void Camera::setTarget(const Eigen::Vector3f& target)
	{
		_changedView = true;
		_useTarget = true;
		_useDirection = false;
		_target = target;
	}

	void Camera::setDirection(const Eigen::Vector3f& direction)
	{
		_changedView = true;
		_useTarget = false;
		_useDirection = true;
		_direction = direction.normalized();
	}

	const Eigen::Vector3f& Camera::target() const
	{
		if (_useTarget && !_useDirection)
		{
			return _target;
		} else if (_useDirection && !_useTarget)
		{
			_target = _position + _direction;
			return _target;
		} else
		{
			VclDebugError("Invalid state");
		}

		return _target;
	}

	const Eigen::Vector3f& Camera::direction() const
	{
		if (_useDirection && !_useTarget)
		{
			return _direction;
		} else if (_useTarget && !_useDirection)
		{
			_direction = (_target - _position).normalized();
		} else
		{
			VclDebugError("Invalid state");
		}

		return _direction;
	}

	void Camera::setUp(const Eigen::Vector3f& up)
	{
		_changedView = true;
		_up = up;
	}

	void Camera::setViewport(int x, int y)
	{
		_changedProjection = true;

		_viewportX = x;
		_viewportY = y;
	}

	void Camera::setFieldOfView(float angle)
	{
		_changedProjection = true;

		_fov = angle;
	}

	void Camera::setFarPlane(float far_plane)
	{
		_changedProjection = true;

		_farPlane = far_plane;
	}

	void Camera::setNearPlane(float near_plane)
	{
		_changedProjection = true;

		_nearPlane = near_plane;
	}

	void Camera::setZoom(float zoom)
	{
		VclRequire(_zoom >= 1.0f, "Zoom greater equal to 1");

		_changedProjection = true;

		_zoom = zoom;
	}

	Eigen::ParametrizedLine3f Camera::viewRay() const
	{
		return Eigen::ParametrizedLine3f::Through(_position, _target);
	}

	const Eigen::Matrix<float, 4, 4, Eigen::DontAlign | Eigen::ColMajor>& Camera::projection() const
	{
		if (_changedProjection)
		{
			_projection = 
				_factory->createPerspectiveFov
				(
					_nearPlane, _farPlane,
					(float) _viewportX / (float) _viewportY,
					_fov / nonLinearZoom(),
					Handedness::RightHanded
				);
			_changedProjection = false;
		}

		return _projection;
	}

	const Eigen::Matrix<float, 4, 4, Eigen::DontAlign | Eigen::ColMajor>& Camera::view() const
	{
		if (_changedView)
		{
			_view =
				_factory->createLookAt
				(
					position(),
					direction(),
					up(),
					Handedness::RightHanded
				);
			_changedView = false;
		}

		return _view;
	}

	void Camera::setView(const Eigen::Matrix4f& view)
	{
		using namespace Eigen;

		// Extract parameters, assume RH
		//Vector3f p = _target + view.col(2).segment<3>(0) * (_position - _target).norm();
		//_position = p;
		//_changedView = true;

		VclDebugError("To do.");

		_view = view;
	}

	void Camera::encloseInFrustum(const Eigen::Vector3f& center, const Eigen::Vector3f& dir_to_camera, float radius, const Eigen::Vector3f& up)
	{
		assert(radius > 0.0f);

		setPosition(center + dir_to_camera.normalized() * 2.0f * 1.05f * radius);
		setTarget(center);
		setUp(up);

		_nearPlane = 0.001f * (position() - target()).norm();
		_farPlane = 3.0f * 1.05f * radius;
		_zoom = 1.0f;

		_changedView = true;
		_changedProjection = true;
	}

	Eigen::ParametrizedLine3f Camera::pickWorldSpace(int x, int y) const
	{
		Eigen::Matrix4f unproject = (projection() * view()).inverse();
		return genericPick(x, y, unproject);
	}

	Eigen::ParametrizedLine3f Camera::pickViewSpace(int x, int y) const
	{
		Eigen::Matrix4f unproject = projection().inverse();
		return genericPick(x, y, unproject);
	}

	Eigen::ParametrizedLine3f Camera::genericPick(int x, int y, Eigen::Matrix4f& unproject) const
	{
		// Convert the mouse point to screen space coordinates
		float ratio_x = (float)x / (float)_viewportX;
		float ratio_y = (float)y / (float)_viewportY;

		Eigen::Vector4f v;
		v.x() = 2.0f * ratio_x - 1.0f;
		v.y() = 1.0f - 2.0f * ratio_y;
		v.z() = 0.0f;
		v.w() = 1.0f;

		// Unproject the point on the near plane
		Eigen::Vector4f start = unproject * v;
		start *= 1.0f / start.w();
		Eigen::Vector3f ray_start(start.x(), start.y(), start.z());

		// Unproject the point on the far plane
		v.z() = 1.0f;

		Eigen::Vector4f end = unproject * v;
		end *= 1.0f / end.w();
		Eigen::Vector3f ray_end(end.x(), end.y(), end.z());

		// Create the unprojected ray
		return Eigen::ParametrizedLine3f::Through(ray_start, ray_end);
	}

	float Camera::nonLinearZoom() const
	{
		return std::exp(zoom() - 1.0f);
	}
}}
