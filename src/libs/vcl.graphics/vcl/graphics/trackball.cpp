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

// VCL
#include <vcl/math/math.h>

namespace Vcl { namespace Graphics
{
	void Trackball::reset(const Eigen::Vector3f& up)
	{
		_rotate = false;
		_up = up;
		_lastQuat = Eigen::Quaternionf::Identity();
	}

	void Trackball::startRotate(float ratio_x, float ratio_y, bool right_handed)
	{
		_rotate = true;
		_lastPosition = project(ratio_x, ratio_y, right_handed);
	}

	void Trackball::startRotate(const Eigen::Quaternionf& inital_rotation, float ratio_x, float ratio_y, bool right_handed)
	{
		_rotate = true;
		_lastQuat = inital_rotation;
		_lastPosition = project(ratio_x, ratio_y, right_handed);
	}

	void Trackball::rotate(float ratio_x, float ratio_y, bool right_handed)
	{
		if (!_rotate) return;

		Eigen::Vector3f new_pos = project(ratio_x, ratio_y, right_handed);
		Eigen::Quaternionf new_quat = fromPosition(new_pos);

		_lastPosition = new_pos;
		_lastQuat = new_quat * _lastQuat;
		_lastQuat.normalize();
	}

	void Trackball::endRotate()
	{
		_rotate = false;
	}

	Eigen::Vector3f Trackball::project(float ratio_x, float ratio_y, bool right_handed)
	{
		using Vcl::Mathematics::clamp;

		float x, y;
		Eigen::Vector3f projection;

		// Scale to [0,0] - [2,2]
		x = 2.0f * ratio_x;
		y = 2.0f * ratio_y;

		// Translate 0,0 to center
		x = clamp(x - 1.0f, -1.0f, 1.0f);
		y = clamp(1.0f - y, -1.0f, 1.0f);

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

		if (!right_handed)
			projection.z() *= -1;

		// Rotate to actual up vector
		auto rot = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f{0.0f, 1.0f, 0.0f}, _up);
		return rot * projection;
	}

	Eigen::Quaternionf Trackball::fromPosition(Eigen::Vector3f v) const
	{
		if (v == _lastPosition)
		{
			return Eigen::Quaternionf::Identity();
		}

		return Eigen::Quaternionf::FromTwoVectors(_lastPosition, v);

		//Eigen::Vector3f d = v - _lastPosition;
		//float t = d.norm() / (2 * _radius);
		//
		//if (t > 1.0f) t = 1.0f;
		//if (t < -1.0f) t = -1.0f;
		//float phi = 2.0f * asin(t);
		//
		//Eigen::Vector3f axis = v.cross(_lastPosition);
		//Eigen::Vector3f weighted_axis = axis.normalized()*sin(phi/2.0f);
		//return{ cos(phi / 2.0f), weighted_axis.x(), weighted_axis.y(), weighted_axis.z() };
	}
}}
