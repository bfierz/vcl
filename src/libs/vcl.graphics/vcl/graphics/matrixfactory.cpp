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
#include <vcl/graphics/matrixfactory.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Graphics
{
	Eigen::Matrix4f MatrixFactory::createLookAt
	(
		const Eigen::Vector3f& position,
		const Eigen::Vector3f& direction,
		const Eigen::Vector3f& world_up,
		Handedness handedness
	) const
	{
		VclRequire(Vcl::Mathematics::equal(direction.norm(), 1.f, 1e-4f), "Direction vector is normalized.");
		VclRequire(Vcl::Mathematics::equal(world_up.norm(),  1.f, 1e-4f), "Up vector is normalized.");

		if (handedness == Handedness::RightHanded)
		{
			Eigen::Vector3f right, up, dir;
			dir = direction;
			right = dir.cross(world_up);
			up = right.cross(dir);

			Eigen::Matrix4f matrix;
			matrix << right.x(), right.y(), right.z(), -right.dot(position),
				         up.x(),    up.y(),    up.z(),    -up.dot(position),
				       -dir.x(),  -dir.y(),  -dir.z(),    dir.dot(position),
				           0.0f,      0.0f,      0.0f,                 1.0f;

			return matrix;
		}
		else if (handedness == Handedness::LeftHanded)
		{
			Eigen::Vector3f right, up, dir;
			dir = direction;
			right = world_up.cross(dir);
			up = dir.cross(right);

			Eigen::Matrix4f matrix;
			matrix << right.x(), right.y(), right.z(), -right.dot(position),
				         up.x(),    up.y(),    up.z(),    -up.dot(position),
				        dir.x(),   dir.y(),   dir.z(),   -dir.dot(position),
				           0.0f,      0.0f,      0.0f,                 1.0f;

			return matrix;
		}
		else
		{
			VclDebugError("Not implemented.");
			return Eigen::Matrix4f::Zero();
		}
	}
}}

namespace Vcl { namespace Graphics { namespace OpenGL
{
	Eigen::Matrix4f MatrixFactory::createPerspective
	(
		float width,
		float height,
		float near_plane,
		float far_plane,
		Handedness handedness
	) const
	{
		VCL_UNREFERENCED_PARAMETER(width);
		VCL_UNREFERENCED_PARAMETER(height);
		VCL_UNREFERENCED_PARAMETER(near_plane);
		VCL_UNREFERENCED_PARAMETER(far_plane);

		if (handedness == Handedness::RightHanded)
		{
		}
		else if (handedness == Handedness::LeftHanded)
		{
		}
		else
		{
			VclDebugError("Not implemented.");
		}

		VclDebugError("Not implemented.");
		return Eigen::Matrix4f::Identity();
	}

	Eigen::Matrix4f MatrixFactory::createPerspectiveFov
	(
		float near_plane,
		float far_plane,
		float aspect_ratio,
		float fov_vertical,
		Handedness handedness
	) const
	{
		if (handedness == Handedness::RightHanded)
		{
			float h = 1.0f / tan(fov_vertical / 2.0f);
			float w = h / aspect_ratio;

			// z device coordinates [0, 1]
			//float _33 = far_plane / (near_plane - far_plane);
			//float _34 = near_plane * _33;

			// z device coordinates [-1, 1]
			float _33 = (near_plane + far_plane) / (near_plane - far_plane);
			float _34 = (2 * near_plane * far_plane) / (near_plane - far_plane);
			
			Eigen::Matrix4f matrix;
			matrix << w, 0, 0, 0,
					  0, h, 0, 0,
					  0, 0, _33, _34,
					  0, 0, -1, 0;
			
			return matrix;
		}
		else if (handedness == Handedness::LeftHanded)
		{
			float h = 1.0f / tan(fov_vertical / 2.0f);
			float w = h / aspect_ratio;

			// z device coordinates [0, 1]
			//float _33 = far_plane / (far_plane - near_plane);
			//float _34 = -near_plane * _33;

			// z device coordinates [-1, 1]
			float _33 = (near_plane + far_plane) / (far_plane - near_plane);
			float _34 = -(2 * near_plane * far_plane) / (far_plane - near_plane);
			
			Eigen::Matrix4f matrix;
			matrix << w, 0, 0, 0,
					  0, h, 0, 0,
					  0, 0, _33, _34,
					  0, 0, 1, 0;
			
			return matrix;
		}
		else
		{
			VclDebugError("Not implemented.");
			return Eigen::Matrix4f::Identity();
		}
	}

	Eigen::Matrix4f MatrixFactory::createPerspectiveOffCenter
	(
		float left,
		float right,
		float bottom,
		float top,
		float near_plane,
		float far_plane,
		Handedness handedness
	) const
	{
		VCL_UNREFERENCED_PARAMETER(left);
		VCL_UNREFERENCED_PARAMETER(right);
		VCL_UNREFERENCED_PARAMETER(bottom);
		VCL_UNREFERENCED_PARAMETER(top);
		VCL_UNREFERENCED_PARAMETER(near_plane);
		VCL_UNREFERENCED_PARAMETER(far_plane);

		if (handedness == Handedness::RightHanded)
		{
		}
		else if (handedness == Handedness::LeftHanded)
		{
		}
		else
		{
			VclDebugError("Not implemented.");
		}

		VclDebugError("Not implemented.");
		return Eigen::Matrix4f::Identity();
	}

	Eigen::Matrix4f MatrixFactory::createOrtho
	(
		float width,
		float height,
		float near_plane,
		float far_plane,
		Handedness handedness
	) const
	{
		if (handedness == Handedness::RightHanded)
		{
			float h = height;
			float w = width;

			// z device coordinates [0, 1]
			//float _33 = 1 / (near_plane - far_plane);
			//float _34 = near_plane * _33;

			// z device coordinates [-1, 1]
			float _33 = 2 / (near_plane - far_plane);
			float _34 = (near_plane + far_plane) / (near_plane - far_plane);
			
			Eigen::Matrix4f matrix;
			matrix << 2/w, 0, 0, 0,
					  0, 2/h, 0, 0,
					  0, 0, _33, _34,
					  0, 0, 0, 1;
			
			return matrix;
		}
		else if (handedness == Handedness::LeftHanded)
		{
			float h = height;
			float w = width;

			// z device coordinates [0, 1]
			//float _33 = 1 / (far_plane - near_plane);
			//float _34 = -near_plane * _33;

			// z device coordinates [-1, 1]
			float _33 = 2 / (far_plane - near_plane);
			float _34 = -(near_plane + far_plane) / (far_plane - near_plane);
			
			Eigen::Matrix4f matrix;
			matrix << 2/w, 0, 0, 0,
					  0, 2/h, 0, 0,
					  0, 0, _33, _34,
					  0, 0, 0, 1;
			
			return matrix;
		}
		else
		{
			VclDebugError("Not implemented.");
			return Eigen::Matrix4f::Identity();
		}
	}

	Eigen::Matrix4f MatrixFactory::createOrthoOffCenter
	(
		float left,
		float right,
		float bottom,
		float top,
		float near_plane,
		float far_plane,
		Handedness handedness
	) const
	{
		VCL_UNREFERENCED_PARAMETER(left);
		VCL_UNREFERENCED_PARAMETER(right);
		VCL_UNREFERENCED_PARAMETER(bottom);
		VCL_UNREFERENCED_PARAMETER(top);
		VCL_UNREFERENCED_PARAMETER(near_plane);
		VCL_UNREFERENCED_PARAMETER(far_plane);

		if (handedness == Handedness::RightHanded)
		{
		}
		else if (handedness == Handedness::LeftHanded)
		{
		}
		else
		{
			VclDebugError("Not implemented.");
		}

		VclDebugError("Not implemented.");
		return Eigen::Matrix4f::Identity();
	}
}}}

namespace Vcl { namespace Graphics { namespace Direct3D
{
	Eigen::Matrix4f MatrixFactory::createPerspective
	(
		float width,
		float height,
		float near_plane,
		float far_plane,
		Handedness handedness
	) const
	{
		VCL_UNREFERENCED_PARAMETER(width);
		VCL_UNREFERENCED_PARAMETER(height);
		VCL_UNREFERENCED_PARAMETER(near_plane);
		VCL_UNREFERENCED_PARAMETER(far_plane);

		if (handedness == Handedness::RightHanded)
		{
		}
		else if (handedness == Handedness::LeftHanded)
		{
		}
		else
		{
			VclDebugError("Not implemented.");
		}

		VclDebugError("Not implemented.");
		return Eigen::Matrix4f::Identity();
	}

	Eigen::Matrix4f MatrixFactory::createPerspectiveFov
	(
		float near_plane,
		float far_plane,
		float aspect_ratio,
		float fov_vertical,
		Handedness handedness
	) const
	{
		if (handedness == Handedness::RightHanded)
		{
			float h = 1.0f / tan(fov_vertical / 2.0f);
			float w = h / aspect_ratio;

			// z device coordinates [0, 1]
			float _33 = far_plane / (near_plane - far_plane);
			float _34 = near_plane * _33;

			// z device coordinates [-1, 1]
			//float _33 = (near_plane + far_plane) / (near_plane - far_plane);
			//float _34 = (2 * near_plane * far_plane) / (near_plane - far_plane);
			
			Eigen::Matrix4f matrix;
			matrix << w, 0, 0, 0,
					  0, h, 0, 0,
					  0, 0, _33, _34,
					  0, 0, -1, 0;
			
			return matrix;
		}
		else if (handedness == Handedness::LeftHanded)
		{
			float h = 1.0f / tan(fov_vertical / 2.0f);
			float w = h / aspect_ratio;

			// z device coordinates [0, 1]
			float _33 = far_plane / (far_plane - near_plane);
			float _34 = -near_plane * _33;

			// z device coordinates [-1, 1]
			//float _33 = (near_plane + far_plane) / (far_plane - near_plane);
			//float _34 = -(2 * near_plane * far_plane) / (far_plane - near_plane);
			
			Eigen::Matrix4f matrix;
			matrix << w, 0, 0, 0,
					  0, h, 0, 0,
					  0, 0, _33, _34,
					  0, 0, 1, 0;
			
			return matrix;
		}
		else
		{
			VclDebugError("Not implemented.");
			return Eigen::Matrix4f::Identity();
		}
	}

	Eigen::Matrix4f MatrixFactory::createPerspectiveOffCenter
	(
		float left,
		float right,
		float bottom,
		float top,
		float near_plane,
		float far_plane,
		Handedness handedness
	) const
	{
		VCL_UNREFERENCED_PARAMETER(left);
		VCL_UNREFERENCED_PARAMETER(right);
		VCL_UNREFERENCED_PARAMETER(bottom);
		VCL_UNREFERENCED_PARAMETER(top);
		VCL_UNREFERENCED_PARAMETER(near_plane);
		VCL_UNREFERENCED_PARAMETER(far_plane);

		if (handedness == Handedness::RightHanded)
		{
		}
		else if (handedness == Handedness::LeftHanded)
		{
		}
		else
		{
			VclDebugError("Not implemented.");
		}

		VclDebugError("Not implemented.");
		return Eigen::Matrix4f::Identity();
	}

	Eigen::Matrix4f MatrixFactory::createOrtho
	(
		float width,
		float height,
		float near_plane,
		float far_plane,
		Handedness handedness
	) const
	{
		if (handedness == Handedness::RightHanded)
		{
			float h = height;
			float w = width;

			// z device coordinates [0, 1]
			float _33 = 1 / (near_plane - far_plane);
			float _34 = near_plane * _33;

			// z device coordinates [-1, 1]
			//float _33 = 2 / (near_plane - far_plane);
			//float _34 = (near_plane + far_plane) / (near_plane - far_plane);
			
			Eigen::Matrix4f matrix;
			matrix << 2/w, 0, 0, 0,
					  0, 2/h, 0, 0,
					  0, 0, _33, _34,
					  0, 0, 0, 1;
			
			return matrix;
		}
		else if (handedness == Handedness::LeftHanded)
		{
			float h = height;
			float w = width;

			// z device coordinates [0, 1]
			float _33 = 1 / (far_plane - near_plane);
			float _34 = -near_plane * _33;

			// z device coordinates [-1, 1]
			//float _33 = 2 / (far_plane - near_plane);
			//float _34 = -(near_plane + far_plane) / (far_plane - near_plane);
			
			Eigen::Matrix4f matrix;
			matrix << 2/w, 0, 0, 0,
					  0, 2/h, 0, 0,
					  0, 0, _33, _34,
					  0, 0, 0, 1;
			
			return matrix;
		}
		else
		{
			VclDebugError("Not implemented.");
			return Eigen::Matrix4f::Identity();
		}
	}

	Eigen::Matrix4f MatrixFactory::createOrthoOffCenter
	(
		float left,
		float right,
		float bottom,
		float top,
		float near_plane,
		float far_plane,
		Handedness handedness
	) const
	{
		VCL_UNREFERENCED_PARAMETER(left);
		VCL_UNREFERENCED_PARAMETER(right);
		VCL_UNREFERENCED_PARAMETER(bottom);
		VCL_UNREFERENCED_PARAMETER(top);
		VCL_UNREFERENCED_PARAMETER(near_plane);
		VCL_UNREFERENCED_PARAMETER(far_plane);

		if (handedness == Handedness::RightHanded)
		{
		}
		else if (handedness == Handedness::LeftHanded)
		{
		}
		else
		{
			VclDebugError("Not implemented.");
		}

		VclDebugError("Not implemented.");
		return Eigen::Matrix4f::Identity();
	}
}}}
