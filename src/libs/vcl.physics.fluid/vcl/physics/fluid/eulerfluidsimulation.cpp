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
#include <vcl/physics/fluid/eulerfluidsimulation.h>

// VCL
#include <vcl/core/contract.h>
#include <vcl/physics/fluid/centergrid.h>

namespace Vcl { namespace Physics { namespace Fluid
{
	EulerFluidSimulation::EulerFluidSimulation()
	{

	}
	EulerFluidSimulation::~EulerFluidSimulation()
	{

	}

	void EulerFluidSimulation::update(Fluid::CenterGrid& g, float dt)
	{
	}

	void EulerFluidSimulation::addBuoyancy(float* forceZ, float buoyancy, const float* source, const Eigen::Vector3i& dim)
	{
		using namespace Eigen;

		if (buoyancy == 0.0f)
			return;

		size_t size = dim.x()*dim.y()*dim.z();
		Map<VectorXf>(forceZ, size) += buoyancy * Map<const VectorXf>(source, size);
	}

	void EulerFluidSimulation::setBorderZero(float* field, const Eigen::Vector3i& dim)
	{
		setBorderZeroX(field, dim);
		setBorderZeroY(field, dim);
		setBorderZeroZ(field, dim);
	}
	void EulerFluidSimulation::setBorderZeroX(float* field, const Eigen::Vector3i& dim)
	{
		const size_t slabSize = dim.x() * dim.y();
		size_t index;
		for (size_t z = 0; z < dim.z(); z++)
		{
			for (size_t y = 0; y < dim.y(); y++)
			{
				// left slab
				index = y * dim.x() + z * slabSize;
				field[index] = 0.0f;

				// right slab
				index += dim.x() - 1;
				field[index] = 0.0f;
			}
		}
	}
	void EulerFluidSimulation::setBorderZeroY(float* field, const Eigen::Vector3i& dim)
	{
		const size_t slabSize = dim.x() * dim.y();
		size_t index;
		for (size_t z = 0; z < dim.z(); z++)
		{
			for (size_t x = 0; x < dim.x(); x++)
			{
				// bottom slab
				index = x + z * slabSize;
				field[index] = 0.0f;

				// top slab
				index += slabSize - dim.x();
				field[index] = 0.0f;
			}
		}
	}
	void EulerFluidSimulation::setBorderZeroZ(float* field, const Eigen::Vector3i& dim)
	{
		const size_t slabSize = dim.x() * dim.y();
		const size_t totalCells = dim.x() * dim.y() * dim.z();
		size_t index;
		for (size_t y = 0; y < dim.y(); y++)
		{
			for (size_t x = 0; x < dim.x(); x++)
			{
				// front slab
				index = x + y * dim.x();
				field[index] = 0.0f;

				// back slab
				index += totalCells - slabSize;
				field[index] = 0.0f;
			}
		}
	}

	void EulerFluidSimulation::setNeumannX(float* field, const Eigen::Vector3i& dim)
	{
		const size_t slab_size = dim.x() * dim.y();
		size_t index;
		for (size_t z = 0; z < dim.z(); z++)
		{
			for (size_t y = 0; y < dim.y(); y++)
			{
				// left slab
				index = y * dim.x() + z * slab_size;
				field[index] = field[index + 2];

				// right slab
				index += dim.x() - 1;
				field[index] = field[index - 2];
			}
		}
	}
	void EulerFluidSimulation::setNeumannY(float* field, const Eigen::Vector3i& dim)
	{
		const size_t slab_size = dim.x() * dim.y();
		size_t index;
		for (size_t z = 0; z < dim.z(); z++)
		{
			for (size_t x = 0; x < dim.x(); x++)
			{
				// bottom slab
				index = x + z * slab_size;
				field[index] = field[index + 2 * dim.x()];

				// top slab
				index += slab_size - dim.x();
				field[index] = field[index - 2 * dim.x()];
			}
		}
	}
	void EulerFluidSimulation::setNeumannZ(float* field, const Eigen::Vector3i& dim)
	{
		const size_t slab_size = dim.x() * dim.y();
		const size_t totalCells = dim.x() * dim.y() * dim.z();
		size_t index;
		for (size_t y = 0; y < dim.y(); y++)
		{
			for (size_t x = 0; x < dim.x(); x++)
			{
				// front slab
				index = x + y * dim.x();
				field[index] = field[index + 2 * slab_size];

				// back slab
				index += totalCells - slab_size;
				field[index] = field[index - 2 * slab_size];
			}
		}
	}

	void EulerFluidSimulation::copyBorderX(float* field, const Eigen::Vector3i& dim)
	{
		const size_t slab_size = dim.x() * dim.y();
		size_t index;
		for (size_t z = 0; z < dim.z(); z++)
		{
			for (size_t y = 0; y <dim.y(); y++)
			{
				// left slab
				index = y * dim.x() + z * slab_size;
				field[index] = field[index + 1];

				// right slab
				index += dim.x() - 1;
				field[index] = field[index - 1];
			}
		}
	}

	void EulerFluidSimulation::copyBorderY(float* field, const Eigen::Vector3i& dim)
	{
		const size_t slab_size = dim.x() * dim.y();
		size_t index;
		for (size_t z = 0; z < dim.z(); z++)
		{
			for (size_t x = 0; x < dim.x(); x++)
			{
				// bottom slab
				index = x + z * slab_size;
				field[index] = field[index + dim.x()];
				// top slab
				index += slab_size - dim.x();
				field[index] = field[index - dim.x()];
			}
		}
	}

	void EulerFluidSimulation::copyBorderZ(float* field, const Eigen::Vector3i& dim)
	{
		const size_t slab_size = dim.x() * dim.y();
		const size_t totalCells = dim.x() * dim.y() * dim.z();
		size_t index;
		for (size_t y = 0; y < dim.y(); y++)
		{
			for (size_t x = 0; x < dim.x(); x++)
			{
				// front slab
				index = x + y * dim.x();
				field[index] = field[index + slab_size];
				// back slab
				index += totalCells - slab_size;
				field[index] = field[index - slab_size];
			}
		}
	}
}}}
