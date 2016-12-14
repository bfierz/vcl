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
	template<typename Real>
	void makePoissonStencil
	(
		unsigned int dim,
		Real h,
		Real a,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ac,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ax_l,
		Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>> Ax_r,
		Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip
	)
	{
		// Scaling of the stencil
		const Real s = a / (h*h);

		Ac.setZero();
		Ax_l.setZero();
		Ax_r.setZero();

		for (typename Eigen::Vector2ui::Scalar i = 1; i < dim - 1; i++)
		{
			// Initialize write-back data
			float a_c   = 0;
			float a_x_l = 0;
			float a_x_r = 0;

			const typename Eigen::Vector2ui::Scalar index = i;
			if (!skip[index])
			{
				if (i < (dim - 1) && !skip[index + 1])
				{
					a_c   -= 1;
					a_x_r  = 1;
				}
				if (i > 0 && !skip[index - 1])
				{
					a_c   -= 1;
					a_x_l  = 1;
				}
			}

			Ac  [index] = s * a_c;
			Ax_l[index] = s * a_x_l;
			Ax_r[index] = s * a_x_r;
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
		Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip
	)
	{
		// Scaling of the stencil
		const Real s = a / (h*h);

		Ac.setZero();
		Ax_l.setZero();
		Ax_r.setZero();
		Ay_l.setZero();
		Ay_r.setZero();

		for (typename Eigen::Vector2ui::Scalar j = 1; j < dim.y() - 1; j++)
		{
			for (typename Eigen::Vector2ui::Scalar i = 1; i < dim.x() - 1; i++)
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
					if (i < (dim.x() - 1) && !skip[index + 1])
					{
						a_c  -= 1;
						a_x_r = 1;
					}
					if (i > 0 && !skip[index - 1])
					{
						a_c  -= 1;
						a_x_l = 1;
					}

					if (j < (dim.y() - 1) && !skip[index + dim.x()])
					{
						a_c  -= 1;
						a_y_r = 1;
					}
					if (j > 0 && !skip[index - dim.x()])
					{
						a_c  -= 1;
						a_y_l = 1;
					}
				}

				Ac  [index] = s * a_c;
				Ax_l[index] = s * a_x_l;
				Ax_r[index] = s * a_x_r;
				Ay_l[index] = s * a_y_l;
				Ay_r[index] = s * a_y_r;
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
		Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>> skip
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

		for (typename Eigen::Vector2ui::Scalar k = 1; k < dim.z() - 1; k++)
		{
			for (typename Eigen::Vector2ui::Scalar j = 1; j < dim.y() - 1; j++)
			{
				for (typename Eigen::Vector2ui::Scalar i = 1; i < dim.x() - 1; i++)
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
						if (i < (dim.x() - 1) && !skip[index + 1])
						{
							a_c  -= 1;
							a_x_r = 1;
						}
						if (i > 0 && !skip[index - 1])
						{
							a_c  -= 1;
							a_x_l = 1;
						}

						if (j < (dim.y() - 1) && !skip[index + dim.x()])
						{
							a_c  -= 1;
							a_y_r = 1;
						}
						if (j > 0 && !skip[index - dim.x()])
						{
							a_c  -= 1;
							a_y_l = 1;
						}

						if (k < (dim.z() - 1) && !skip[index + slab])
						{
							a_c  -= 1;
							a_z_r = 1;
						}
						if (k > 0 && !skip[index - slab])
						{
							a_c  -= 1;
							a_z_l = 1;
						}
					}

					Ac  [index] = s * a_c;
					Ax_l[index] = s * a_x_l;
					Ax_r[index] = s * a_x_r;
					Ay_l[index] = s * a_y_l;
					Ay_r[index] = s * a_y_r;
					Az_l[index] = s * a_z_l;
					Az_r[index] = s * a_z_r;
				}
			}
		}
	}
}}}
