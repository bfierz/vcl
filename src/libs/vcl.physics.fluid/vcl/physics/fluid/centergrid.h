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
#include <array>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Physics { namespace Fluid
{
	template<int BlockSize>
	class GridIndex3D
	{
	public:
		inline GridIndex3D(int x, int y, int z)
		: _x(x)
		, _y(y)
		, _z(z)
		{
		}

		inline int operator() (int x, int y, int z)
		{
			const int blockSize = BlockSize*BlockSize*BlockSize;

			int W = _x / BlockSize;
			int H = _y / BlockSize;
			int D = _z / BlockSize;

			int X = x / BlockSize;
			int Y = y / BlockSize;
			int Z = z / BlockSize;

			x %= BlockSize;
			y %= BlockSize;
			z %= BlockSize;

			int bIdx = X + Y*W + Z*W*H;
			return blockSize*bIdx + x + y*BlockSize + z*BlockSize*BlockSize;
		}

	private:
		int _x;
		int _y;
		int _z;
	};

	class CenterGrid
	{
	public:
		CenterGrid(const Eigen::Vector3i& resolution, float spacing, int blockSize);
		virtual ~CenterGrid();

	public:
		const Eigen::Vector3i& resolution() const { return _resolution; }
		float spacing() const { return _spacing; }

		//! Returns the size of the data blocks
		int blockSize() const { return _blockSize;  }

		float buoyancy() const { return _buoyancy; }
		void setBuoyancy(float b) { _buoyancy = b; }

		float heatDiffusion() const { return _heatDiffusion; }

		float vorticityCoeff() const { return _vorticityCoeff; }
		void setVorticityCoeff(float v) { _vorticityCoeff = v; }

	public:
		virtual void resize(const Eigen::Vector3i& resolution) = 0;

		virtual void swap() = 0;

	private:
		Eigen::Vector3i _resolution;
		float _spacing;

		//! Internal block size
		int _blockSize;

		//! Buoyancy in the fluid volume
		float _buoyancy = 0;

		//! Heat diffusion constant
		float _heatDiffusion = 0;

		//! Scaling of the vorticity
		float _vorticityCoeff = 0;
	};

	class DefaultCenterGrid : public CenterGrid
	{
	public:
		DefaultCenterGrid(const Eigen::Vector3i& resolution, float spacing);
		virtual ~DefaultCenterGrid();

	public:
		Eigen::VectorXf& forces(int dim);
		Eigen::VectorXf& velocities(int i, int dim);

		Eigen::VectorXi& obstacles();

		Eigen::VectorXf& densities(int i);

		Eigen::VectorXf& heat(int i);

	public:
		virtual void resize(const Eigen::Vector3i& resolution) override;

		virtual void swap() override;

	public:
		virtual uint32_t* aquireObstacles();
		virtual float*    aquireVelocityX();
		virtual float*    aquireVelocityY();
		virtual float*    aquireVelocityZ();

		virtual void releaseObstacles();
		virtual void releaseVelocityX();
		virtual void releaseVelocityY();
		virtual void releaseVelocityZ();

	private:
		std::array<Eigen::VectorXf, 3> _force;

		std::array<std::array<Eigen::VectorXf, 3>, 2> _velocity;

		Eigen::VectorXf                _vorticityMag;
		std::array<Eigen::VectorXf, 3> _vorticity;

		Eigen::VectorXi                _obstacles;
		std::array<Eigen::VectorXf, 3> _obstaclesVelocity;

		std::array<Eigen::VectorXf, 2> _density;

		std::array<Eigen::VectorXf, 2> _heat;

	};
}}}
