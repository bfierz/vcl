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

namespace Vcl { namespace Graphics {
	enum class Handedness
	{
		LeftHanded = 0,
		RightHanded = 1
	};

	class MatrixFactory
	{
	public:
		Eigen::Matrix4f createLookAt
		(
			const Eigen::Vector3f& position,
			const Eigen::Vector3f& direction,
			const Eigen::Vector3f& world_up,
			Handedness handedness = Handedness::RightHanded
		) const;

	public:
		virtual Eigen::Matrix4f createPerspective
		(
			float width, float height, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
		) const = 0;

		virtual Eigen::Matrix4f createPerspectiveFov
		(
			float near_plane, float far_plane, float aspect_ratio, float fov_vertical, Handedness handedness = Handedness::RightHanded
		) const = 0;

		virtual Eigen::Matrix4f createPerspectiveOffCenter
		(
			float left, float right, float bottom, float top, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
		) const = 0;

		virtual Eigen::Matrix4f createOrtho
		(
			float width, float height, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
		) const = 0;

		virtual Eigen::Matrix4f createOrthoOffCenter
		(
			float left, float right, float bottom, float top, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
		) const = 0;
	};

	namespace OpenGL
	{
		class MatrixFactory : public Vcl::Graphics::MatrixFactory
		{
		public:
			virtual Eigen::Matrix4f createPerspective
			(
				float width, float height, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
			) const override;

			virtual Eigen::Matrix4f createPerspectiveFov
			(
				float near_plane, float far_plane, float aspect_ratio, float fov_vertical, Handedness handedness = Handedness::RightHanded
			) const override;

			virtual Eigen::Matrix4f createPerspectiveOffCenter
			(
				float left, float right, float bottom, float top, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
			) const override;

			virtual Eigen::Matrix4f createOrtho
			(
				float width, float height, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
			) const override;

			virtual Eigen::Matrix4f createOrthoOffCenter
			(
				float left, float right, float bottom, float top, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
			) const override;
		};
	}
	
	namespace Direct3D
	{
		class MatrixFactory : public Vcl::Graphics::MatrixFactory
		{
		public:
			virtual Eigen::Matrix4f createPerspective
			(
				float width, float height, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
			) const override;

			virtual Eigen::Matrix4f createPerspectiveFov
			(
				float near_plane, float far_plane, float aspect_ratio, float fov_vertical, Handedness handedness = Handedness::RightHanded
			) const override;

			virtual Eigen::Matrix4f createPerspectiveOffCenter
			(
				float left, float right, float bottom, float top, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
			) const override;

			virtual Eigen::Matrix4f createOrtho
			(
				float width, float height, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
			) const override;

			virtual Eigen::Matrix4f createOrthoOffCenter
			(
				float left, float right, float bottom, float top, float near_plane, float far_plane, Handedness handedness = Handedness::RightHanded
			) const override;
		};
	}
}}
