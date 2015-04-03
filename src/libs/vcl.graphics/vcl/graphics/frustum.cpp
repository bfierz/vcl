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
#include <vcl/graphics/frustum.h>

// Eigen library
#include <Eigen/Eigenvalues>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/matrixfactory.h>
#include <vcl/graphics/camera.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Graphics
{
	template<typename Scalar>
	PerspectiveViewFrustum<Scalar>::PerspectiveViewFrustum()
	: PerspectiveViewFrustum(0, 0, (float) (M_PI / 4.0), 0.01, 100, {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0})
	{
	}

	template<typename Scalar>
	PerspectiveViewFrustum<Scalar>::PerspectiveViewFrustum
	(
		real_t width, real_t height, real_t fov, real_t near_plane, real_t far_plane,
		vector3_t pos, vector3_t dir, vector3_t up, vector3_t right
	)
	:	_x(width), _y(height), _fov(fov), _near(near_plane), _far(far_plane),
		_position(pos), _direction(dir), _up(up), _right(right)
	{
		computePlanes();
	}

	template<typename Scalar>
	PerspectiveViewFrustum<Scalar>::PerspectiveViewFrustum(const Vcl::Graphics::Camera* cam)
	: PerspectiveViewFrustum
	  (
		cam->viewportWidth(), cam->viewportHeight(), cam->fieldOfView(), cam->nearPlane(), cam->farPlane(),
		cam->position().cast<Scalar>(),
		cam->direction().cast<Scalar>(),
		cam->direction().cross(cam->up().cross(cam->direction()).normalized()).normalized().cast<Scalar>(),
		cam->up().cross(cam->direction()).normalized().cast<Scalar>()
	  )
	{
	}

	template<typename Scalar>
	PerspectiveViewFrustum<Scalar>::PerspectiveViewFrustum(const PerspectiveViewFrustum<real_t>& rhs)
	{
		_x = rhs._x;
		_y = rhs._y;
		_fov = rhs._fov;
		_near = rhs._near;
		_far = rhs._far;
		_position = rhs._position;
		_direction = rhs._direction;
		_up = rhs._up;
		_right = rhs._right;

		_corners = rhs._corners;
		_planes = rhs._planes;
	}

	template<typename Scalar>
	const typename PerspectiveViewFrustum<Scalar>::vector3_t& PerspectiveViewFrustum<Scalar>::position() const
	{
		return _position;
	}
					
		
	template<typename Scalar>
	const typename PerspectiveViewFrustum<Scalar>::vector3_t& PerspectiveViewFrustum<Scalar>::direction() const
	{
		return _direction;
	}
		
	template<typename Scalar>
	const typename PerspectiveViewFrustum<Scalar>::vector3_t& PerspectiveViewFrustum<Scalar>::up() const
	{
		return _up;
	}	
		
	template<typename Scalar>
	const typename PerspectiveViewFrustum<Scalar>::vector3_t& PerspectiveViewFrustum<Scalar>::right() const
	{
		return _right;
	}

	template<typename Scalar>
	typename PerspectiveViewFrustum<Scalar>::real_t PerspectiveViewFrustum<Scalar>::width() const
	{
		return _x;
	}
	
	template<typename Scalar>
	typename PerspectiveViewFrustum<Scalar>::real_t PerspectiveViewFrustum<Scalar>::height() const
	{
		return _y;
	}

	template<typename Scalar>
	typename PerspectiveViewFrustum<Scalar>::real_t PerspectiveViewFrustum<Scalar>::fieldOfView() const
	{
		return _fov;
	}
	
	template<typename Scalar>
	typename PerspectiveViewFrustum<Scalar>::real_t PerspectiveViewFrustum<Scalar>::nearPlane() const
	{
		return _near;
	}
		
	template<typename Scalar>
	typename PerspectiveViewFrustum<Scalar>::real_t PerspectiveViewFrustum<Scalar>::farPlane() const
	{
		return _far;
	}

	template<typename Scalar>
	bool PerspectiveViewFrustum<Scalar>::isInside(vector3_t p)
	{
		for (int i = 0; i < 6; i++)
			if (_planes[i].signedDistance(p) < 0)
				return false; 

		return true;
	}

	template<typename Scalar>
	const typename PerspectiveViewFrustum<Scalar>::vector3_t& PerspectiveViewFrustum<Scalar>::corner(unsigned int i) const
	{
		Require(i < 8, "Index is valid.");

		return _corners[i];
	}

	template<typename Scalar>
	void PerspectiveViewFrustum<Scalar>::computePlanes()
	{
		using Vcl::Mathematics::equal;

		Require(equal(_direction.norm(), 1, (Scalar) 1e-6), "Direction is unit length.", "Length: %f", _direction.norm());
		Require(equal(_up.norm(), 1, (Scalar) 1e-6), "Up is unit length.", "Length: %f", _up.norm());
		Require(equal(_right.norm(), 1, (Scalar) 1e-6), "Right is unit length.", "Length: %f", _right.norm());
			
		real_t ratio = _x / _y;

		real_t near_height = real_t(2) * std::tan(real_t(0.5) * _fov) * _near;
		real_t near_width = near_height * ratio;
			
		real_t far_height = real_t(2) * std::tan(real_t(0.5) * _fov) * _far;
		real_t far_width = far_height * ratio;

		// Far plane center
		vector3_t fc = _position + _direction * _far;

		// Far plane (top, left/top, right/bottom, left/top, left)
		vector3_t ftl = fc + (_up * far_height*real_t(0.5)) - (_right * far_width*real_t(0.5));
		vector3_t ftr = fc + (_up * far_height*real_t(0.5)) + (_right * far_width*real_t(0.5));
		vector3_t fbl = fc - (_up * far_height*real_t(0.5)) - (_right * far_width*real_t(0.5));
		vector3_t fbr = fc - (_up * far_height*real_t(0.5)) + (_right * far_width*real_t(0.5));
			
		// Near plane center
		vector3_t nc = _position + _direction * _near;
			
		// Near plane (top, left/top, right/bottom, left/top, left)
		vector3_t ntl = nc + (_up * near_height*real_t(0.5)) - (_right * near_width*real_t(0.5));
		vector3_t ntr = nc + (_up * near_height*real_t(0.5)) + (_right * near_width*real_t(0.5));
		vector3_t nbl = nc - (_up * near_height*real_t(0.5)) - (_right * near_width*real_t(0.5));
		vector3_t nbr = nc - (_up * near_height*real_t(0.5)) + (_right * near_width*real_t(0.5));
			
		// Store the corners
		_corners[0] = nbl;
		_corners[1] = nbr;
		_corners[2] = ntr;
		_corners[3] = ntl;

		_corners[4] = fbl;
		_corners[5] = fbr;
		_corners[6] = ftr;
		_corners[7] = ftl;

		// Compute the bounding planes
		//vector3_t aux, normal;
		//
		//aux = (nc + _y*near_height) - _position;
		//aux.normalize();
		//normal = aux * _x;
		//_planes[Top] = Eigen::Hyperplane<real_t, 3>(normal, nc+_up*near_height);
		//
		//aux = (nc - _y*near_height) - _position;
		//aux.normalize();
		//normal = _x * aux;
		//_planes[Bottom] = Eigen::Hyperplane<real_t, 3>(normal, nc-_y*near_height);
		//
		//aux = (nc - _x*far_width) - _position;
		//aux.normalize();
		//normal = aux * _y;
		//_planes[Left] = Eigen::Hyperplane<real_t, 3>(normal, nc-_right*near_width);
		//
		//aux = (nc + _x*far_width) - _position;
		//aux.normalize();
		//normal = _y * aux;
		//_planes[Right] = Eigen::Hyperplane<real_t, 3>(normal, nc+_right*near_width);
		//
		//_planes[Near] = Eigen::Hyperplane<real_t, 3>(-_direction, nc);
		//_planes[Far] = Eigen::Hyperplane<real_t, 3>(_direction, fc);
	}

	template<typename Scalar>
	OrthographicViewFrustum<Scalar>::OrthographicViewFrustum()
	: _x(0)
	, _y(0)
	, _near(0.01)
	, _far(100)
	, _position(0, 0, 0)
	, _direction(0, 0, 1)
	, _up(0, 1, 0)
	, _right(1, 0, 0)
	{
		computePlanes();
	}

	template<typename Scalar>
	OrthographicViewFrustum<Scalar>::OrthographicViewFrustum
	(
		real_t width, real_t height, real_t near_plane, real_t far_plane, vector3_t pos, vector3_t dir, vector3_t up, vector3_t right
	)
	: _x(width)
	, _y(height)
	, _near(near_plane)
	, _far(far_plane)
	, _position(pos)
	, _direction(dir)
	, _up(up)
	, _right(right)
	{
		computePlanes();
	}

	template<typename Scalar>
	OrthographicViewFrustum<Scalar>::OrthographicViewFrustum
	(
		const OrthographicViewFrustum<real_t>& rhs
	)
	{
		_x = rhs._x;
		_y = rhs._y;
		_near = rhs._near;
		_far = rhs._far;
		_position = rhs._position;
		_direction = rhs._direction;
		_up = rhs._up;
		_right = rhs._right;

		_corners = rhs._corners;
		_planes = rhs._planes;
	}

	template<typename Scalar>
	const typename OrthographicViewFrustum<Scalar>::vector3_t& OrthographicViewFrustum<Scalar>::position() const
	{
		return _position;
	}
					
	template<typename Scalar>
	const typename OrthographicViewFrustum<Scalar>::vector3_t& OrthographicViewFrustum<Scalar>::direction() const
	{
		return _direction;
	}
		
	template<typename Scalar>
	const typename OrthographicViewFrustum<Scalar>::vector3_t& OrthographicViewFrustum<Scalar>::up() const
	{
		return _up;
	}
		
	template<typename Scalar>
	const typename OrthographicViewFrustum<Scalar>::vector3_t& OrthographicViewFrustum<Scalar>::right() const
	{
		return _right;
	}
			
		
	template<typename Scalar>
	typename OrthographicViewFrustum<Scalar>::real_t OrthographicViewFrustum<Scalar>::nearPlane() const
	{
		return _near;
	}
		
	template<typename Scalar>
	typename OrthographicViewFrustum<Scalar>::real_t OrthographicViewFrustum<Scalar>::farPlane() const
	{
		return _far;
	}
					
	template<typename Scalar>
	typename OrthographicViewFrustum<Scalar>::real_t OrthographicViewFrustum<Scalar>::width() const
	{
		return _x;
	}
					
	template<typename Scalar>
	typename OrthographicViewFrustum<Scalar>::real_t OrthographicViewFrustum<Scalar>::height() const
	{
		return _y;
	}

	template<typename Scalar>
	bool OrthographicViewFrustum<Scalar>::isInside(vector3_t p)
	{
		for (int i = 0; i < 6; i++)
			if (_planes[i].signedDistance(p) < 0)
				return false; 

		return true;
	}

	template<typename Scalar>
	const typename OrthographicViewFrustum<Scalar>::vector3_t& OrthographicViewFrustum<Scalar>::corner(unsigned int i) const
	{
		Require(i < 8, "Index is valid.");

		return _corners[i];
	}

	template<typename Scalar>
	Eigen::Matrix<Scalar, 4, 4> OrthographicViewFrustum<Scalar>::computeViewMatrix(const MatrixFactory& factory) const
	{
		using Vcl::Mathematics::equal;
			
		Require(equal(_direction.cross(_up).dot(_right), 1, (Scalar) 1e-4), "Frame is orthogonal.", "Angle: %f", _direction.cross(_up).dot(_right));

		return factory.createLookAt(_position.cast<float>(), _direction.cast<float>(), _up.cast<float>(), Handedness::RightHanded).cast<Scalar>();
	}

	template<typename Scalar>
	Eigen::Matrix<Scalar, 4, 4> OrthographicViewFrustum<Scalar>::computeProjectionMatrix(const MatrixFactory& factory) const
	{		
		Require(_x > 0, "Width is valid");
		Require(_y > 0, "Height is valid");
		Require(_near > 0, "Near plane is valid");
		Require(_far > 0, "Far plane is valid");
		
		return factory.createOrtho((float) _x, (float) _y, (float) nearPlane(), (float) farPlane(), Handedness::RightHanded).cast<Scalar>();
	}
	
	template<typename Scalar>
	OrthographicViewFrustum<Scalar> OrthographicViewFrustum<Scalar>::enclose(const PerspectiveViewFrustum<real_t>& frustum, const vector3_t& orthographic_direction)
	{
		using Vcl::Mathematics::equal;

		vector3_t dir = -orthographic_direction.normalized();

		// OBB positions & normals
		std::array<vector3_t, 6> p; p.fill(frustum.corner(0));
		std::array<vector3_t, 6> n;
		std::array<int, 6> idx;     idx.fill(0);

		// Near/Far plane
		n[0] =  dir;
		n[1] = -dir;
		for (int i = 1; i < 8; i++)
		{
			real_t d0 = n[0].dot(frustum.corner(i) - p[0]);
			real_t d1 = n[1].dot(frustum.corner(i) - p[1]);
			if (d0 > 0) { p[0] = frustum.corner(i); idx[0] = i; }
			if (d1 > 0) { p[1] = frustum.corner(i); idx[1] = i; }
		}

		// Project points on near plane
		std::array<vector3_t, 8> proj_points;
		for (int i = 0; i < 8; i++)
		{
			real_t d = dir.dot(frustum.corner(i) - p[0]);
			proj_points[i] = frustum.corner(i) - d * dir;
			Check(equal(dir.dot(proj_points[i] - p[0]), 0, (Scalar) 1e-3), "Projected point is on plane", "d = %f", dir.dot(proj_points[i] - p[0]));
		}

		// Compute center of projected points
		vector3_t m = vector3_t::Zero();
		for (int i = 0; i < 8; i++)
			m += proj_points[i];
		m /= 8;

		// Compute projected center point
		//vector3_t c = 
		//	real_t(0.5)*(frustum.position() + frustum.nearPlane()*frustum.direction()) + 
		//	real_t(0.5)*(frustum.position() +  frustum.farPlane()*frustum.direction())  ;
		//real_t dc = dir.dot(c - p[0]);
		//c -= dc * dir;

		// Compute PCA of projected points
		Eigen::Matrix<real_t, 3, 8> Y;
		for (int i = 0; i < 8; i++)
			Y.col(i) = proj_points[i] - m;

		Eigen::Matrix<real_t, 3, 3> S = Y*Y.transpose();
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix<real_t, 3, 3>> eig(S);

		// Compute planes orthogonal to the main directions
		n[2] =  eig.eigenvectors().col(2).normalized();
		n[3] = -eig.eigenvectors().col(2).normalized();
		n[4] =  eig.eigenvectors().col(1).normalized();
		n[5] = -eig.eigenvectors().col(1).normalized();

		// Check if frame is orthogonal, else change it
		if (-dir.cross(n[5]).dot(n[3]) < 0)
			std::swap(n[4], n[5]);
			
		for (int i = 1; i < 8; i++)
		{
			real_t d2 = n[2].dot(frustum.corner(i) - p[2]);
			real_t d3 = n[3].dot(frustum.corner(i) - p[3]);
			real_t d4 = n[4].dot(frustum.corner(i) - p[4]);
			real_t d5 = n[5].dot(frustum.corner(i) - p[5]);
			if (d2 > 0) { p[2] = frustum.corner(i); idx[2] = i; }
			if (d3 > 0) { p[3] = frustum.corner(i); idx[3] = i; }
			if (d4 > 0) { p[4] = frustum.corner(i); idx[4] = i; }
			if (d5 > 0) { p[5] = frustum.corner(i); idx[5] = i; }
		}

		// Compute frustum parameters
		real_t near_to_far   = n[1].dot(p[1] - p[0]);
		real_t left_to_right = n[3].dot(p[3] - p[2]);
		real_t bottom_to_top = n[5].dot(p[5] - p[4]);

		// Compute frustum center on near plane
		// - Project m on to left plane, displace by left_to_right/2 -> t
		// - Project t on to bottom plane, displace by bottom_to_top/2 -> nc
		real_t dt = -n[3].dot(m - p[3]) - left_to_right/2;
		vector3_t t = m + dt * n[3];
		real_t dnc = -n[5].dot(t - p[5]) - bottom_to_top/2;
		vector3_t nc = t + dnc * n[5];

		AssertBlock
		{
			vector3_t pos = nc + dir*near_to_far;
			real_t d0 = n[0].dot(pos - p[0]);
			real_t d1 = n[1].dot(pos - p[1]);

			real_t dm0 = n[0].dot(nc - p[0]);
			real_t dm1 = n[1].dot(nc - p[1]);
			real_t dm2 = n[2].dot(nc - p[2]);
			real_t dm3 = n[3].dot(nc - p[3]);
			real_t dm4 = n[4].dot(nc - p[4]);
			real_t dm5 = n[5].dot(nc - p[5]);

			Check(equal(abs(d0),   near_to_far, (Scalar) 1e-3), "Frustum position is correct.");
			Check(equal(abs(d1), 2*near_to_far, (Scalar) 1e-3), "Frustum position is correct.");
				
			Check(equal(abs(dm0),           0, (Scalar) 1e-3), "Frustum depth is correct.");
			Check(equal(abs(dm1), near_to_far, (Scalar) 1e-3), "Frustum depth is correct.");
			Check(equal(abs(dm2), left_to_right/2, (Scalar) 1e-3), "Frustum width is correct.");
			Check(equal(abs(dm3), left_to_right/2, (Scalar) 1e-3), "Frustum width is correct.");
			Check(equal(abs(dm4), bottom_to_top/2, (Scalar) 1e-3), "Frustum height is correct.");
			Check(equal(abs(dm5), bottom_to_top/2, (Scalar) 1e-3), "Frustum height is correct.");

			Check(equal(-dir.cross(n[5]).dot(n[3]), 1, (Scalar) 1e-4), "Frame is orthogonal.", "Angle: %f", -dir.cross(n[5]).dot(n[3]));
		}

		OrthographicViewFrustum<real_t> ortho
		(
			left_to_right, bottom_to_top, near_to_far, 2*near_to_far, nc + dir*near_to_far, -dir, n[5], n[3]
		);
		return ortho;
	}
	
	template<typename Scalar>
	void OrthographicViewFrustum<Scalar>::computePlanes()
	{
		using Vcl::Mathematics::equal;

		Require(equal(_direction.squaredNorm(), 1, (Scalar) 1e-6), "Direction is unit length.");
		Require(equal(_up.squaredNorm(), 1, (Scalar) 1e-6), "Up is unit length.");
		Require(equal(_right.squaredNorm(), 1, (Scalar) 1e-6), "Right is unit length.");
			
		real_t ratio = _x / _y;

		real_t height = _y;
		real_t width  = _x;

		// Far plane center
		vector3_t fc = _position + _direction * _far;

		// Far plane (top, left/top, right/bottom, left/top, left)
		vector3_t ftl = fc + (_up * height*real_t(0.5)) - (_right * width*real_t(0.5));
		vector3_t ftr = fc + (_up * height*real_t(0.5)) + (_right * width*real_t(0.5));
		vector3_t fbl = fc - (_up * height*real_t(0.5)) - (_right * width*real_t(0.5));
		vector3_t fbr = fc - (_up * height*real_t(0.5)) + (_right * width*real_t(0.5));
			
		// Near plane center
		vector3_t nc = _position + _direction * _near;
			
		// Near plane (top, left/top, right/bottom, left/top, left)
		vector3_t ntl = nc + (_up * height*real_t(0.5)) - (_right * width*real_t(0.5));
		vector3_t ntr = nc + (_up * height*real_t(0.5)) + (_right * width*real_t(0.5));
		vector3_t nbl = nc - (_up * height*real_t(0.5)) - (_right * width*real_t(0.5));
		vector3_t nbr = nc - (_up * height*real_t(0.5)) + (_right * width*real_t(0.5));
			
		// Store the corners
		_corners[0] = nbl;
		_corners[1] = nbr;
		_corners[2] = ntr;
		_corners[3] = ntl;

		_corners[4] = fbl;
		_corners[5] = fbr;
		_corners[6] = ftr;
		_corners[7] = ftl;

		// Compute the bounding planes
		//vector3_t aux, normal;
		//
		//aux = (nc + _y*near_height) - _position;
		//aux.normalize();
		//normal = aux * _x;
		//_planes[Top] = Eigen::Hyperplane<real_t, 3>(normal, nc+_up*near_height);
		//
		//aux = (nc - _y*near_height) - _position;
		//aux.normalize();
		//normal = _x * aux;
		//_planes[Bottom] = Eigen::Hyperplane<real_t, 3>(normal, nc-_y*near_height);
		//
		//aux = (nc - _x*far_width) - _position;
		//aux.normalize();
		//normal = aux * _y;
		//_planes[Left] = Eigen::Hyperplane<real_t, 3>(normal, nc-_right*near_width);
		//
		//aux = (nc + _x*far_width) - _position;
		//aux.normalize();
		//normal = _y * aux;
		//_planes[Right] = Eigen::Hyperplane<real_t, 3>(normal, nc+_right*near_width);
		//
		//_planes[Near] = Eigen::Hyperplane<real_t, 3>(-_direction, nc);
		//_planes[Far] = Eigen::Hyperplane<real_t, 3>(_direction, fc);
	}

	template class PerspectiveViewFrustum<float>;
	template class PerspectiveViewFrustum<double>;

	template class OrthographicViewFrustum<float>;
	template class OrthographicViewFrustum<double>;
}}
