/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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

// VCL
#include <vcl/core/contract.h>
#include <vcl/physics/fluid/centergrid.h>

namespace Vcl { namespace Physics { namespace Fluid
{
	template<typename T>
	class Advection
	{
	public:
		Advection()
		:	mX(0), mY(0), mZ(0)
		{
		}

		Advection(unsigned int x, unsigned int y, unsigned int z)
		:	mX(x), mY(y), mZ(z)
		{
		}

		virtual ~Advection()
		{
		}

	public:
		virtual void setSize(unsigned int x, unsigned int y, unsigned int z)
		{
			mX = x;
			mY = y;
			mZ = z;
		}

		unsigned int x() const { return mX; }
		unsigned int y() const { return mY; }
		unsigned int z() const { return mZ; }

	public:
		virtual void operator() (T dt, const T* vel_x, const T* vel_y, const T* vel_z, const T* in_field, T* out_field) = 0;

	private: /* Grid Size */
		unsigned int mX;
		unsigned int mY;
		unsigned int mZ;
	};

	template<typename T>
	class SemiLagrangeAdvection : public Advection<T>
	{
	public:
		SemiLagrangeAdvection()
		:	Advection<T>()
		{
		}

		SemiLagrangeAdvection(unsigned int x, unsigned int y, unsigned int z)
		:	Advection<T>(x, y, z)
		{
		}

		virtual ~SemiLagrangeAdvection()
		{
		}

	public:
		virtual void operator() (T dt, const T* vel_x, const T* vel_y, const T* vel_z, const T* old_field, T* new_field) override
		{
			using namespace Eigen;

			VclRequire(vel_x != NULL, "");
			VclRequire(vel_y != NULL, "");
			VclRequire(vel_z != NULL, "");

			VclRequire(x() > 0, "");
			VclRequire(y() > 0, "");
			VclRequire(z() > 0, "");

			VclRequire(x() % 4 == 0, "");

			for (size_t z = 0; z < this->z(); z++)
			{
				for (size_t y = 0; y < this->y(); y++)
				{
					for (size_t x = 0; x < this->x(); x++)
					{
						const size_t index = x + y * this->x() + z * this->x()*this->y();

						// backtrace
						float xTrace = x - dt * vel_x[index];
						float yTrace = y - dt * vel_y[index];
						float zTrace = z - dt * vel_z[index];

						// clamp backtrace to grid boundaries
						if (xTrace < 0.5f) xTrace = 0.5f;
						if (xTrace > this->x() - 1.5f) xTrace = this->x() - 1.5f;
						if (yTrace < 0.5f) yTrace = 0.5f;
						if (yTrace > this->y() - 1.5f) yTrace = this->y() - 1.5f;
						if (zTrace < 0.5f) zTrace = 0.5f;
						if (zTrace > this->z() - 1.5f) zTrace = this->z() - 1.5f;

						// locate neighbors to interpolate
						const size_t x0 = (size_t)xTrace;
						const size_t x1 = x0 + 1;
						const size_t y0 = (size_t)yTrace;
						const size_t y1 = y0 + 1;
						const size_t z0 = (size_t)zTrace;
						const size_t z1 = z0 + 1;

						// get interpolation weights
						const float s1 = xTrace - x0;
						const float s0 = 1.0f - s1;
						const float t1 = yTrace - y0;
						const float t0 = 1.0f - t1;
						const float u1 = zTrace - z0;
						const float u0 = 1.0f - u1;

						const size_t i000 = x0 + y0 * this->x() + z0 * this->x()*this->y();
						const size_t i010 = x0 + y1 * this->x() + z0 * this->x()*this->y();
						const size_t i100 = x1 + y0 * this->x() + z0 * this->x()*this->y();
						const size_t i110 = x1 + y1 * this->x() + z0 * this->x()*this->y();
						const size_t i001 = x0 + y0 * this->x() + z1 * this->x()*this->y();
						const size_t i011 = x0 + y1 * this->x() + z1 * this->x()*this->y();
						const size_t i101 = x1 + y0 * this->x() + z1 * this->x()*this->y();
						const size_t i111 = x1 + y1 * this->x() + z1 * this->x()*this->y();

						// interpolate
						// (indices could be computed once)
						new_field[index] = u0 * (s0 * (t0 * old_field[i000] +
													   t1 * old_field[i010]) +
												 s1 * (t0 * old_field[i100] +
													   t1 * old_field[i110])) +
										   u1 * (s0 * (t0 * old_field[i001] +
									  				   t1 * old_field[i011]) +
												 s1 * (t0 * old_field[i101] +
													   t1 * old_field[i111]));
					}
				}
			}
		}
	};
	
	template<typename T>
	class MacCormackAdvection : public Advection<T>
	{
	public:
		MacCormackAdvection()
		:	Advection<T>(),
			mSemiLagrangeAdvection()
		{
		}

		MacCormackAdvection(unsigned int x, unsigned int y, unsigned int z)
		:	Advection<T>(x, y, z),
			mSemiLagrangeAdvection(x, y, z)
		{
			mIntermediateField.setZero(this->x()*this->y()*this->z());
		}

		virtual ~MacCormackAdvection()
		{
		}

	public:
		virtual void setSize(unsigned int x, unsigned int y, unsigned int z) override
		{
			Advection<T>::setSize(x, y, z);

			// Configure the semi-lagrange advection
			mSemiLagrangeAdvection.setSize(x, y, z);

			// Resize the intermediate field
			mIntermediateField.setZero(this->x()*this->y()*this->z());
		}

	public:
		virtual void operator() (T dt, const T* vel_x, const T* vel_y, const T* vel_z, const T* old_field, T* new_field) override
		{
			using namespace Eigen;

			VclRequire(vel_x != NULL, "");
			VclRequire(vel_y != NULL, "");
			VclRequire(vel_z != NULL, "");

			VclRequire(x() > 0, "");
			VclRequire(y() > 0, "");
			VclRequire(z() > 0, "");

			const T* phiN = old_field;
			T* phiN1 = new_field;

			T* phiHatN = new_field;
			T* phiHatN1 = mIntermediateField.data();

			// phiHatN1 = A(phiN)
			mSemiLagrangeAdvection(		   dt, vel_x, vel_y, vel_z, phiN, phiHatN1);

			// phiHatN = A^R(phiHatN1)
			mSemiLagrangeAdvection(-1.0f * dt, vel_x, vel_y, vel_z, phiHatN1, phiHatN);

			// phiN1 = phiHatN1 + (phiN - phiHatN) / 2
			unsigned int size = this->x()*this->y()*this->z();
			Map<Matrix<T, Dynamic, 1>>(phiN1, size) =
				Map<Matrix<T, Dynamic, 1>>(phiHatN1, size) +
				0.5 * (Map<Matrix<T, Dynamic, 1>>(phiN, size) - Map<Matrix<T, Dynamic, 1>>(phiHatN, size));

			copyBorderX(phiN1);
			copyBorderY(phiN1);
			copyBorderZ(phiN1);

			// Clamp any newly created extrema
			clampExtrema(dt, vel_x, vel_y, vel_z, phiN, phiN1);

			// If the error estimate was bad, revert to first order
			clampOutsideRays(dt, vel_x, vel_y, vel_z, phiN, phiN1, phiHatN1);
		}

	private:
		void copyBorderX(T* field)
		{
			const size_t slabSize = this->x() * this->y();
			size_t index;
			for (size_t z = 0; z < this->z(); z++)
			{
				for (size_t y = 0; y < this->y(); y++)
				{
					// left slab
					index = y * this->x() + z * slabSize;
					field[index] = field[index + 1];

					// right slab
					index += this->x() - 1;
					field[index] = field[index - 1];
				}
			}
		}
	
		void copyBorderY(T* field)
		{
			const size_t slabSize = this->x() * this->y();
			size_t index;
			for (size_t z = 0; z < this->z(); z++)
			{
				for (size_t x = 0; x < this->x(); x++)
				{
					// bottom slab
					index = x + z * slabSize;
					field[index] = field[index + this->x()]; 
					// top slab
					index += slabSize - this->x();
					field[index] = field[index - this->x()];
				}
			}
		}
	
		void copyBorderZ(T* field)
		{
			const size_t slabSize = this->x() * this->y();
			const size_t totalCells = this->x() * this->y() * this->z();
			size_t index;
			for (size_t y = 0; y < this->y(); y++)
			{
				for (size_t x = 0; x < this->x(); x++)
				{
					// front slab
					index = x + y * this->x();
					field[index] = field[index + slabSize]; 
					// back slab
					index += totalCells - slabSize;
					field[index] = field[index - slabSize];
				}
			}
		}
		
		void clampExtrema
		(
			const T dt,
			const T* velx,
			const T* vely,
			const T* velz,
			const T* old_field,
			float* new_field
		)
		{
			const size_t xres = this->x();
			const size_t yres = this->y();
			const size_t zres = this->z();
			const size_t slabSize = xres * yres;
		
			size_t index = xres*yres + xres + 1;
			for (size_t z = 1; z < zres - 1; z++, index += 2 * xres)
			{
				for (size_t y = 1; y < yres - 1; y++, index += 2)
				{
					for (size_t x = 1; x < xres - 1; x++, index++)
					{
						// backtrace
						T xTrace = x - dt * velx[index];
						T yTrace = y - dt * vely[index];
						T zTrace = z - dt * velz[index];

						// clamp backtrace to grid boundaries
						if (xTrace < 0.5f) xTrace = 0.5f;
						if (xTrace > xres - 1.5f) xTrace = xres - 1.5f;
						if (yTrace < 0.5f) yTrace = 0.5f;
						if (yTrace > yres - 1.5f) yTrace = yres - 1.5f;
						if (zTrace < 0.5f) zTrace = 0.5f;
						if (zTrace > zres - 1.5f) zTrace = zres - 1.5f;

						// locate neighbors to interpolate
						const size_t x0 = (size_t)xTrace;
						const size_t x1 = x0 + 1;
						const size_t y0 = (size_t)yTrace;
						const size_t y1 = y0 + 1;
						const size_t z0 = (size_t)zTrace;
						const size_t z1 = z0 + 1;

						const size_t i000 = x0 + y0 * xres + z0 * slabSize;
						const size_t i010 = x0 + y1 * xres + z0 * slabSize;
						const size_t i100 = x1 + y0 * xres + z0 * slabSize;
						const size_t i110 = x1 + y1 * xres + z0 * slabSize;
						const size_t i001 = x0 + y0 * xres + z1 * slabSize;
						const size_t i011 = x0 + y1 * xres + z1 * slabSize;
						const size_t i101 = x1 + y0 * xres + z1 * slabSize;
						const size_t i111 = x1 + y1 * xres + z1 * slabSize;

						T minField = old_field[i000];
						T maxField = old_field[i000];

						minField = (old_field[i010] < minField) ? old_field[i010] : minField;
						maxField = (old_field[i010] > maxField) ? old_field[i010] : maxField;

						minField = (old_field[i100] < minField) ? old_field[i100] : minField;
						maxField = (old_field[i100] > maxField) ? old_field[i100] : maxField;

						minField = (old_field[i110] < minField) ? old_field[i110] : minField;
						maxField = (old_field[i110] > maxField) ? old_field[i110] : maxField;

						minField = (old_field[i001] < minField) ? old_field[i001] : minField;
						maxField = (old_field[i001] > maxField) ? old_field[i001] : maxField;

						minField = (old_field[i011] < minField) ? old_field[i011] : minField;
						maxField = (old_field[i011] > maxField) ? old_field[i011] : maxField;

						minField = (old_field[i101] < minField) ? old_field[i101] : minField;
						maxField = (old_field[i101] > maxField) ? old_field[i101] : maxField;

						minField = (old_field[i111] < minField) ? old_field[i111] : minField;
						maxField = (old_field[i111] > maxField) ? old_field[i111] : maxField;

						new_field[index] = (new_field[index] > maxField) ? maxField : new_field[index];
						new_field[index] = (new_field[index] < minField) ? minField : new_field[index];
					}
				}
			}
		}

		void clampOutsideRays
		(
			const T dt,
			const T* velx, const T* vely, const T* velz,
			const T* old_field, T* new_field,
			const T* old_advection
		)
		{
			const size_t sx = this->x();
			const size_t sy = this->y();
			const size_t sz = this->z();

			size_t index = sx*sy + sx + 1;
			for (size_t z = 1; z < sz - 1; z++, index += 2 * sx)
			{
				for (size_t y = 1; y < sy - 1; y++, index += 2)
				{
					for (size_t x = 1; x < sx - 1; x++, index++)
					{
						// backtrace
						T xBackward = x + dt * velx[index];
						T yBackward = y + dt * vely[index];
						T zBackward = z + dt * velz[index];
						T xTrace    = x - dt * velx[index];
						T yTrace    = y - dt * vely[index];
						T zTrace    = z - dt * velz[index];

						// see if it goes outside the boundaries
						bool hasObstacle = 
							(zTrace < 1.0f)    || (zTrace > sz - 2.0f) ||
							(yTrace < 1.0f)    || (yTrace > sy - 2.0f) ||
							(xTrace < 1.0f)    || (xTrace > sx - 2.0f) ||
							(zBackward < 1.0f) || (zBackward > sz - 2.0f) ||
							(yBackward < 1.0f) || (yBackward > sy - 2.0f) ||
							(xBackward < 1.0f) || (xBackward > sx - 2.0f);
						// reuse old advection instead of doing another one...
						if(hasObstacle)
						{
							new_field[index] = old_advection[index];
							continue;
						}
					}
				}
			}
		}

	private:
		SemiLagrangeAdvection<T> mSemiLagrangeAdvection;
		Eigen::Matrix<T, Eigen::Dynamic, 1> mIntermediateField;
	};
}}}
