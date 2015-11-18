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
#include <vcl/graphics/trackball.h>

namespace Vcl { namespace Graphics
{
	void Trackball::reset()
	{
		_rotate = false;
		_lastQuat = Eigen::Quaternionf::Identity();
	}

	void Trackball::startRotate(float ratio_x, float ratio_y, bool right_handed)
	{
		_rotate = true;
		_lastPosition = project(ratio_x, ratio_y);
		if (!right_handed)
			_lastPosition.z() *= -1;
	}

	void Trackball::rotate(float ratio_x, float ratio_y, bool right_handed)
	{
		if (!_rotate) return;

		Eigen::Vector3f new_pos = project(ratio_x, ratio_y);
		if (!right_handed)
			new_pos.z() *= -1;

		Eigen::Quaternionf new_quat = fromPosition(new_pos);

		_lastPosition = new_pos;
		_lastQuat = new_quat * _lastQuat;
		_lastQuat.normalize();
	}

	void Trackball::endRotate()
	{
		_rotate = false;
	}

	Eigen::Vector3f Trackball::project(float ratio_x, float ratio_y)
	{
		float x, y;
		Eigen::Vector3f projection;

		// Scale to [0,0] - [2,2]
		x = 2.0f * ratio_x;
		y = 2.0f * ratio_y;

		// Translate 0,0 to center
		x = x - 1.0f;
		y = 1.0f - y;

		if (x > 1.0f) x = 1.0f;
		if (x < -1.0f) x = -1.0f;
		if (y > 1.0f) y = 1.0f;
		if (y < -1.0f) y = -1.0f;

		// Computation of position to back-project from Bell's Arc-ball
		projection.x() = x;
		projection.y() = y;
		float d = sqrt(x * x + y * y);
		if (d < _radius * 0.70710678118654752440f)
		{
			projection.z() = sqrt(_radius * _radius - d * d);
		}
		else
		{
			float t = _radius * 0.70710678118654752440f;
			projection.z() = t * t / d;
		}

		return projection;
	}

	Eigen::Quaternionf Trackball::fromPosition(Eigen::Vector3f v) const
	{
		if (v == _lastPosition)
		{
			return Eigen::Quaternionf::Identity();
		}

		Eigen::Vector3f axis = _lastPosition.cross(v);
		float phi = atan2f(axis.norm(), _lastPosition.dot(v));

		Eigen::Vector3f weighted_axis = axis.normalized()*sin(phi/2.0f);
		return{ cos(phi / 2.0f), weighted_axis.x(), weighted_axis.y(), weighted_axis.z() };
	}
}}
