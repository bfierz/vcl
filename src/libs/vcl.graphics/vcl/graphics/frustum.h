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
#include <memory>

namespace Vcl { namespace Graphics
{
	// Forward declaration
	class Camera;
	class MatrixFactory;

	template<typename Scalar>
	class PerspectiveViewFrustum
	{
	public: // Typedefs
		using real_t = Scalar;
		using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	public:
		enum
		{
			Top = 0, Bottom, Left, Right, Near, Far
		};

	public:
		//! Default constructor.
		PerspectiveViewFrustum();

		//! Constructor.
		PerspectiveViewFrustum
		(
			real_t width, real_t height, real_t fov, real_t near_plane, real_t far_plane, vector3_t pos, vector3_t dir, vector3_t up, vector3_t right
		);

		//! Construct frustum from camera
		PerspectiveViewFrustum(const Vcl::Graphics::Camera* cam);

		//! Copy constructor
		/*!
		 *	\param rhs object to be copied
		 */
		PerspectiveViewFrustum(const PerspectiveViewFrustum<real_t>& rhs);

	public:
		const vector3_t& position() const;

		const vector3_t& direction() const;

		const vector3_t& up() const;

		const vector3_t& right() const;

		real_t width() const;

		real_t height() const;

		real_t fieldOfView() const;

		real_t nearPlane() const;

		real_t farPlane() const;

	public:
		bool isInside(vector3_t p);

		const vector3_t& corner(unsigned int i) const;

	private:
		void computePlanes();

	private:
		real_t _x, _y;
		real_t _fov;
		real_t _near, _far;
		vector3_t _position, _direction, _up, _right;

		// Corners
		std::array<vector3_t, 8> _corners;

		// Bounding planes
		std::array<Eigen::Hyperplane<real_t, 3>, 6> _planes;
	};

	template<typename Scalar>
	class OrthographicViewFrustum
	{
	public: // Typedefs
		using real_t = Scalar;
		using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	public:
		enum
		{
			Top = 0, Bottom, Left, Right, Near, Far
		};

	public:
		//! Default constructor.
		OrthographicViewFrustum();

		//! Constructor.
		OrthographicViewFrustum
		(
			real_t width, real_t height, real_t near_plane, real_t far_plane, vector3_t pos, vector3_t dir, vector3_t up, vector3_t right
		);

		//! Copy constructor
		/*!
		 *	\param rhs object to be copied
		 */
		OrthographicViewFrustum
		(
			const OrthographicViewFrustum<real_t>& rhs
		);

	public:
		const vector3_t& position() const;

		const vector3_t& direction() const;

		const vector3_t& up() const;

		const vector3_t& right() const;

		real_t nearPlane() const;

		real_t farPlane() const;

		real_t width() const;

		real_t height() const;

	public:
		bool isInside(vector3_t p);

		const vector3_t& corner(unsigned int i) const;

	public:
		Eigen::Matrix<real_t, 4, 4> computeViewMatrix(const MatrixFactory& factory) const;

		Eigen::Matrix<real_t, 4, 4> computeProjectionMatrix(const MatrixFactory& factory) const;

	public:
		static OrthographicViewFrustum<real_t> enclose(const PerspectiveViewFrustum<real_t>& frustum, const vector3_t& orthographic_direction);

	private:
		void computePlanes();

	private:
		real_t _x, _y;
		real_t _near, _far;
		vector3_t _position, _direction, _up, _right;

		// Corners
		std::array<vector3_t, 8> _corners;

		// Bounding planes
		std::array<Eigen::Hyperplane<real_t, 3>, 6> _planes;
	};

// Extern template specialization
#ifndef VCL_GRAPHICS_FRUSTUM_INST
		extern template class PerspectiveViewFrustum<float>;
		extern template class PerspectiveViewFrustum<double>;

		extern template class OrthographicViewFrustum<float>;
		extern template class OrthographicViewFrustum<double>;
#endif /* VCL_GRAPHICS_FRUSTUM_INST */
}}
