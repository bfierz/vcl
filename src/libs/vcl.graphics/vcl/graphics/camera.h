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
#include <atomic>
#include <memory>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/matrixfactory.h>

namespace Vcl { namespace Graphics
{
	/*!
	 *	\brief	Basic camera, projection a 3D scene to a 2D space.
	 *
	 *	Basic perspective camera, which has implementations for right- and
	 *	left-handed coordinate systems.
	 */
	class Camera
	{
	public:
		/*!	
		 *	\brief Default constructor.
		 *
		 *	\param[in] factory Factory creating transformation matrices.
		 *
		 *	Construct an unpositioned camera with a specific matrix construction factory.
		 */
		Camera(std::shared_ptr<MatrixFactory> factory);

		/*!	
		 *	\brief Construct a camera at a position looking a target point.
		 *
		 *	\param[in] position Position of the camera.
		 *	\param[in] target Point the camera is looking at.
		 *	\param[in] factory Factory creating transformation matrices.
		 *
		 *	Construct a camera at a position looking a target point.
		 */
		Camera(Eigen::Vector3f position, Eigen::Vector3f target, std::shared_ptr<MatrixFactory> factory);

		//! Destructor
		~Camera() = default;

	public:
		//! Access the view ray
		Eigen::ParametrizedLine3f viewRay() const;

		//! Pick a point in the view space
		/*!
		 *	The method computes a line in the view frustum which intersects
		 *	the scene. The method assumes that (0,0) is in the upper left
		 *	corner of the viewport.
		 *
		 *	\param x x coordinate of the viewport
		 *	\param y y coordinate of the viewport
		 */
		Eigen::ParametrizedLine3f pickViewSpace(int x, int y) const;

		//! Pick a point in the world space
		/*!
		 *	The method computes a line in the view frustum which intersects
		 *	the scene. The method assumes that (0,0) is in the upper left
		 *	corner of the viewport.
		 *
		 *	\param x x coordinate of the viewport
		 *	\param y y coordinate of the viewport
		 */
		Eigen::ParametrizedLine3f pickWorldSpace(int x, int y) const;

	public: // Properties

		//! Adjust the view port
		void setViewport(int x, int y);

		//! Width of the view port
		int viewportWidth() const { return _viewportX; }

		//! Height of the view port
		int viewportHeight() const { return _viewportY; }

		//! Set the field of view
		void setFieldOfView(float angle);

		//! Get the field of view
		float fieldOfView() const { return _fov; }

		//! Get the far plane
		float farPlane() const { return _farPlane; }

		//! Set the far plane
		void setFarPlane(float far);
		
		//! Get the near plane
		float nearPlane() const { return _nearPlane; }

		//! Set the near plane
		void setNearPlane(float near);

		//! Set the zoom factor
		void setZoom(float zoom);

		//! Get the zoom factor
		float zoom() const { return _zoom; }

		//! Set camera position
		void setPosition(const Eigen::Vector3f& position);

		//! Get camera position
		const Eigen::Vector3f& position() const { return _position; }

		//! Set look-at position
		void setTarget(const Eigen::Vector3f& target);

		//! Get look-at position
		const Eigen::Vector3f& target() const;
		
		//! Set look-at direction
		void setDirection(const Eigen::Vector3f& direction);

		//! Get look-at direction
		const Eigen::Vector3f& direction() const;

		//! Set up vector
		void setUp(const Eigen::Vector3f& up);

		//! Get up vector
		const Eigen::Vector3f& up() const { return _up; }

		//! Get the projection matrix
		const Eigen::Matrix<float, 4, 4, Eigen::DontAlign | Eigen::ColMajor>& projection() const;

		//! Get the view matrix
		const Eigen::Matrix<float, 4, 4, Eigen::DontAlign | Eigen::ColMajor>& view() const;
		
		//! Set camera parameters from a view matrix
		void setView(const Eigen::Matrix4f& view);

	public:
		//! Set the camera parameters such that the defined part of the scene is in the frustum
		void encloseInFrustum(const Eigen::Vector3f& center, const Eigen::Vector3f& dir_to_camera, float radius, const Eigen::Vector3f& up = { 0, 1, 0 });

	private:
		//! Unproject a 2D point
		Eigen::ParametrizedLine3f genericPick(int x, int y, Eigen::Matrix4f& unproject) const;

		//! Calculate the non-linear zoom
		float nonLinearZoom() const;

		//! Matrix factory
		const MatrixFactory& matrixFactory() const { return *_factory; }

	private:
		//! Shared matrix factory instance
		std::shared_ptr<MatrixFactory> _factory;

	private:
		//!	Camera position
		Eigen::Vector3f _position;

		//! Camera look-at point
		mutable Eigen::Vector3f _target;
		bool _useTarget;
		
		//! Camera look-at direction
		mutable Eigen::Vector3f _direction;
		bool _useDirection;

		//! Camera up vector
		Eigen::Vector3f _up;

		//! Field of view
		float _fov;

		//! Distance to the near plane
		float _nearPlane;

		//! Distance to the far plane
		float _farPlane;

		//! Zoom factor
		float _zoom;

		//! Viewport width
		int _viewportX;

		//! Viewport height
		int _viewportY;

		//! Cached view matrix
		mutable Eigen::Matrix<float, 4, 4, Eigen::DontAlign | Eigen::ColMajor> _view;

		//! Cached projection matrix
		mutable Eigen::Matrix<float, 4, 4, Eigen::DontAlign | Eigen::ColMajor> _projection;

		//! Dirty flag: View
		mutable bool _changedView;

		//! Dirty flag: Projection
		mutable bool _changedProjection;
	};
}}
