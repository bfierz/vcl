/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2016 Basil Fierz
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

namespace Vcl { namespace Mathematics { namespace Solver
{
	namespace Detail
	{
		template<typename Real>
		void updateStencil(unsigned int i, unsigned int dim, Real s, Real& c, Real& r, Real& l)
		{
			if (i < (dim - 1))
			{
				c -= s;
				r = s;
			}
			if (i > 0)
			{
				c -= s;
				l = s;
			}
		}
	}

	template<typename Real>
	void makePoissonStencil
	(
		unsigned int dim,
		Real h,
		Real a,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ac,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ax_l,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ax_r,
		Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip
	)
	{
		// Scaling of the stencil
		const Real s = a / (h*h);

		Ac.setZero();
		Ax_l.setZero();
		Ax_r.setZero();

		for (unsigned int i = 0; i < dim; i++)
		{
			// Initialize write-back data
			float a_c   = 0;
			float a_x_l = 0;
			float a_x_r = 0;

			const unsigned int index = i;
			if (!skip[index])
			{
				Detail::updateStencil(i, dim, s, a_c, a_x_r, a_x_l);
			}

			Ac  [index] = a_c;
			Ax_l[index] = a_x_l;
			Ax_r[index] = a_x_r;
		}
	}

	template<typename Real>
	void makePoissonStencil
	(
		Eigen::Vector2ui dim,
		Real h,
		Real a,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ac,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ax_l,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ax_r,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ay_l,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ay_r,
		Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip
	)
	{
		// Scaling of the stencil
		const Real s = a / (h*h);

		Ac.setZero();
		Ax_l.setZero();
		Ax_r.setZero();
		Ay_l.setZero();
		Ay_r.setZero();

		for (typename Eigen::Vector2ui::Scalar j = 0; j < dim.y(); j++)
		{
			for (typename Eigen::Vector2ui::Scalar i = 0; i < dim.x(); i++)
			{
				// Initialize write-back data
				float a_c   = 0;
				float a_x_l = 0;
				float a_x_r = 0;
				float a_y_l = 0;
				float a_y_r = 0;

				const typename Eigen::Vector2ui::Scalar index = j * dim.x() + i;
				if (!skip[index])
				{
					Detail::updateStencil(i, dim.x(), s, a_c, a_x_r, a_x_l);
					Detail::updateStencil(j, dim.y(), s, a_c, a_y_r, a_y_l);
				}

				Ac  [index] = a_c;
				Ax_l[index] = a_x_l;
				Ax_r[index] = a_x_r;
				Ay_l[index] = a_y_l;
				Ay_r[index] = a_y_r;
			}
		}
	}

	template<typename Real>
	void makePoissonStencil
	(
		Eigen::Vector3ui dim,
		Real h,
		Real a,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ac,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ax_l,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ax_r,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ay_l,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ay_r,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Az_l,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Az_r,
		Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip
	)
	{
		// Scaling of the stencil
		const Real s = a / (h*h);

		Ac.setZero();
		Ax_l.setZero();
		Ax_r.setZero();
		Ay_l.setZero();
		Ay_r.setZero();
		Az_l.setZero();
		Az_r.setZero();

		const typename Eigen::Vector2ui::Scalar slab = dim.x()*dim.y();

		for (typename Eigen::Vector2ui::Scalar k = 0; k < dim.z(); k++)
		{
			for (typename Eigen::Vector2ui::Scalar j = 0; j < dim.y(); j++)
			{
				for (typename Eigen::Vector2ui::Scalar i = 0; i < dim.x(); i++)
				{
					// Initialize write-back data
					float a_c   = 0;
					float a_x_l = 0;
					float a_x_r = 0;
					float a_y_l = 0;
					float a_y_r = 0;
					float a_z_l = 0;
					float a_z_r = 0;

					const typename Eigen::Vector2ui::Scalar index = k * slab + j * dim.x() + i;
					if (!skip[index])
					{
						Detail::updateStencil(i, dim.x(), s, a_c, a_x_r, a_x_l);
						Detail::updateStencil(j, dim.y(), s, a_c, a_y_r, a_y_l);
						Detail::updateStencil(k, dim.z(), s, a_c, a_z_r, a_z_l);
					}

					Ac  [index] = a_c;
					Ax_l[index] = a_x_l;
					Ax_r[index] = a_x_r;
					Ay_l[index] = a_y_l;
					Ay_r[index] = a_y_r;
					Az_l[index] = a_z_l;
					Az_r[index] = a_z_r;
				}
			}
		}
	}
}}}
