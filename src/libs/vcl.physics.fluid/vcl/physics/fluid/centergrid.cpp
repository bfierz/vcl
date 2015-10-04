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
#include <vcl/physics/fluid/centergrid.h>

namespace Vcl { namespace Physics { namespace Fluid
{
	CenterGrid::CenterGrid(const Eigen::Vector3i& resolution, float spacing, int blockSize)
	: _resolution(resolution)
	, _spacing(spacing)
	, _blockSize(blockSize)
	{
	}

	CenterGrid::~CenterGrid()
	{
	}

	DefaultCenterGrid::DefaultCenterGrid(const Eigen::Vector3i& resolution, float spacing)
	: CenterGrid(resolution, spacing, -1)
	{

	}
	DefaultCenterGrid::~DefaultCenterGrid()
	{

	}

	void DefaultCenterGrid::resize(const Eigen::Vector3i& resolution)
	{
	}

	void DefaultCenterGrid::swap()
	{

	}

	uint32_t* DefaultCenterGrid::aquireObstacles()
	{
		return (uint32_t*) _obstacles.data();
	}
	float*    DefaultCenterGrid::aquireVelocityX()
	{
		return _velocity[0][0].data();
	}
	float*    DefaultCenterGrid::aquireVelocityY()
	{
		return _velocity[0][1].data();
	}
	float*    DefaultCenterGrid::aquireVelocityZ()
	{
		return _velocity[0][2].data();
	}

	void DefaultCenterGrid::releaseObstacles() {}
	void DefaultCenterGrid::releaseVelocityX() {}
	void DefaultCenterGrid::releaseVelocityY() {}
	void DefaultCenterGrid::releaseVelocityZ() {}
}}}
