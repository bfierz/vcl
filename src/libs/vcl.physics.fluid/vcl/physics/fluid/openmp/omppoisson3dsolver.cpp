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
#include <vcl/physics/fluid/openmp/omppoisson3dsolver.h>

// VCL
#include <vcl/mathematics/solver/eigenconjugategradientscontext.h>
#include <vcl/math/solver/conjugategradients.h>

#include <vcl/physics/fluid/centergrid.h>

#ifdef VCL_OPENMP_SUPPORT
namespace Vcl { namespace Physics { namespace Fluid { namespace OpenMP
{
	CenterGrid3DPoissonCgCtx::CenterGrid3DPoissonCgCtx
	(
		Eigen::Vector3i dim
	)
	: EigenCgBaseContext(nullptr)
	{
	}
	CenterGrid3DPoissonCgCtx::~CenterGrid3DPoissonCgCtx()
	{
	}

	void CenterGrid3DPoissonCgCtx::setData
	(
		std::array<Eigen::VectorXf, 4>* laplacian,
		Eigen::Map<Eigen::VectorXi>* obstacles,
		Eigen::Map<Eigen::VectorXf>* pressure,
		Eigen::Map<Eigen::VectorXf>* divergence,
		Eigen::Vector3i dim
	)
	{
		_laplacian = laplacian;
		_obstacles = obstacles;
		_pressure = pressure;
		_divergence = divergence;
		_dim = dim;

		// Resize the buffers if necessary
		this->resize(dim.x()*dim.y()*dim.z());

		// Set the solution vector
		setX(_pressure);
	}

	void CenterGrid3DPoissonCgCtx::computeInitialResidual()
	{
		float X = _dim.x();
		float Y = _dim.y();
		float Z = _dim.z();

		auto& Ac = (*_laplacian)[0];
		auto& Ax = (*_laplacian)[1];
		auto& Ay = (*_laplacian)[3];
		auto& Az = (*_laplacian)[2];

		auto& field = *_pressure;
		auto& rhs = *_divergence;
		auto& skip = *_obstacles;

		size_t index = X*Y + X + 1;
		for (size_t z = 1; z < Z - 1; z++, index += 2 * X)
		{
			for (size_t y = 1; y < Y - 1; y++, index += 2)
			{
				for (size_t x = 1; x < X - 1; x++, index++)
				{
					float r = rhs[index] -
					(
						field[index] * Ac[index] +
						field[index - 1] * Ax[index] +
						field[index + 1] * Ax[index - 1] +
						field[index - X] * Ay[index] +
						field[index + X] * Ay[index - X] +
						field[index - X*Y] * Az[index] +
						field[index + X*Y] * Az[index - X*Y]
					);
					r = (skip[index]) ? 0.0f : r;
					mResidual[index] = r;
					mDirection[index] = r;
				}
			}
		}
	}

	void CenterGrid3DPoissonCgCtx::computeQ()
	{
		float X = _dim.x();
		float Y = _dim.y();
		float Z = _dim.z();

		auto& Ac = (*_laplacian)[0];
		auto& Ax = (*_laplacian)[1];
		auto& Ay = (*_laplacian)[3];
		auto& Az = (*_laplacian)[2];

		auto& field = mDirection;
		auto& skip = *_obstacles;

		size_t index = X*Y + X + 1;
		for (size_t z = 1; z < Z - 1; z++, index += 2 * X)
		{
			for (size_t y = 1; y < Y - 1; y++, index += 2)
			{
				for (size_t x = 1; x < X - 1; x++, index++)
				{
					float q =
					(
						field[index] * Ac[index] +
						field[index - 1] * Ax[index] +
						field[index + 1] * Ax[index - 1] +
						field[index - X] * Ay[index] +
						field[index + X] * Ay[index - X] +
						field[index - X*Y] * Az[index] +
						field[index + X*Y] * Az[index - X*Y]
					);
					q = (skip[index]) ? 0.0f : q;
					mQ[index] = q;
				}
			}
		}
	}

	CenterGrid3DPoissonSolver::CenterGrid3DPoissonSolver()
	: Fluid::CenterGrid3DPoissonSolver()
	{
		using namespace Eigen;

		// Instantiate the right solver
		_solver = std::make_unique<Vcl::Mathematics::Solver::ConjugateGradients>();
		_solver->setIterationChunkSize(10);
		_solver->setMaxIterations(10);
		_solver->setPrecision(0.0001);

		// Instantiate the 3D Poisson CG context
		_solverCtx = std::make_unique<CenterGrid3DPoissonCgCtx>(Eigen::Vector3i(16, 16, 16));

		// Initialize the buffers
		unsigned int cells = 16 * 16 * 16;
		
		_laplacian[0] = VectorXf::Zero(cells);
		_laplacian[1] = VectorXf::Zero(cells);
		_laplacian[2] = VectorXf::Zero(cells);
		_laplacian[3] = VectorXf::Zero(cells);
		_pressure     = VectorXf::Zero(cells);
		_divergence   = VectorXf::Zero(cells);
	}

	CenterGrid3DPoissonSolver::~CenterGrid3DPoissonSolver()
	{
	}

	void CenterGrid3DPoissonSolver::solve(Fluid::CenterGrid& g)
	{
		Fluid::CenterGrid* grid = &g;
		if (!grid)
			return;

		int x = grid->resolution().x();
		int y = grid->resolution().y();
		int z = grid->resolution().z();
		int cells = x * y * z;
		Check(cells >= 0, "Dimensions are positive.");
		size_t mem_size = cells * sizeof(float);

		// Fetch the internal solver buffers
		auto& laplacian0 = _laplacian[0];
		auto& laplacian1 = _laplacian[1];
		auto& laplacian2 = _laplacian[2];
		auto& laplacian3 = _laplacian[3];

		// Resize the internal solver buffers
		if (laplacian0.size() != cells)
		{
			laplacian0.setZero(cells);
			laplacian1.setZero(cells);
			laplacian2.setZero(cells);
			laplacian3.setZero(cells);
			_pressure.setZero(cells);
			_divergence.setZero(cells);
		}

		// Fetch the grid data
		auto obstacles = Eigen::Map<Eigen::VectorXi>((int*) grid->aquireObstacles(), cells);

		auto vel_x = Eigen::Map<Eigen::VectorXf>(grid->aquireVelocityX(), cells);
		auto vel_y = Eigen::Map<Eigen::VectorXf>(grid->aquireVelocityY(), cells);
		auto vel_z = Eigen::Map<Eigen::VectorXf>(grid->aquireVelocityZ(), cells);

		// Convert the interal data
		auto pressure = Eigen::Map<Eigen::VectorXf>(_pressure.data(), cells);
		auto divergence = Eigen::Map<Eigen::VectorXf>(_divergence.data(), cells);

		// Execute the solver to ensure a divergence free velocity field
		Eigen::Vector3i dim(x, y, z);
		float cellSize = grid->spacing();
		float invCellSize = 1.0f / cellSize;

		_solverCtx->setData
		(
			&_laplacian,
			&obstacles,
			&pressure,
			&divergence,
			grid->resolution()
		);

		// Compute the divergence of the field
		computeDivergence
		(
			&divergence,
			&pressure,
			&vel_x,
			&vel_y,
			&vel_z,
			obstacles,
			grid->resolution(),
			cellSize
		);

		// The original code modified the borders of the pressure values.
		// However this does not help much because the pressure is only use to increase convergence
		//copyBorderX(pressure.data(), dim);
		//copyBorderY(pressure.data(), dim);
		//copyBorderZ(pressure.data(), dim);

		// Build the left hand side of the laplace equation
		buildLHS
		(
			laplacian0,
			laplacian1,
			laplacian2,
			laplacian3,
			obstacles,
			grid->resolution()
		);

		// Compute the pressure field
		_solver->solve(_solverCtx.get());

		// project out solution
		for (int z = 1; z < dim.z() - 1; z++)
		{
			for (int y = 1; y < dim.y() - 1; y++)
			{
				for (int x = 1; x < dim.x() - 1; x++)
				{
					int index = z*dim.x()*dim.y() + y*dim.x() + x;

					vel_x[index] -= 0.5f * (pressure[index + 1]               - pressure[index - 1])               * invCellSize;
					vel_y[index] -= 0.5f * (pressure[index + dim.x()]         - pressure[index - dim.x()])         * invCellSize;
					vel_z[index] -= 0.5f * (pressure[index + dim.x()*dim.y()] - pressure[index - dim.x()*dim.y()]) * invCellSize;
				}
			}
		}

		grid->releaseVelocityZ();
		grid->releaseVelocityY();
		grid->releaseVelocityX();

	}

	void CenterGrid3DPoissonSolver::computeDivergence
	(
		Eigen::Map<Eigen::VectorXf>* divergence,
		Eigen::Map<Eigen::VectorXf>* pressure,
		Eigen::Map<Eigen::VectorXf>* vel_x,
		Eigen::Map<Eigen::VectorXf>* vel_y,
		Eigen::Map<Eigen::VectorXf>* vel_z,
		const Eigen::VectorXi& obstacles,
		Eigen::Vector3i  dim,
		float cell_size
	)
	{
		// Calculate divergence
		for (int z = 1; z < dim.z() - 1; z++)
		{
			for (int y = 1; y < dim.y() - 1; y++)
			{
				for (int x = 1; x < dim.x() - 1; x++)
				{
					int index = z*dim.x()*dim.y() + y*dim.x() + x;

					float xright  = (*vel_x)[index + 1];
					float xleft   = (*vel_x)[index - 1];
					float yup     = (*vel_y)[index + dim.x()];
					float ydown   = (*vel_y)[index - dim.x()];
					float ztop    = (*vel_z)[index + dim.x()*dim.y()];
					float zbottom = (*vel_z)[index - dim.x()*dim.y()];

					/*
					if(obstacles[index + 1]) xright = mObstaclesVelocityX[index + 1];
					if(obstacles[index - 1]) xleft  = mObstaclesVelocityX[index - 1];
					if(obstacles[index + mX]) yup    = mObstaclesVelocityY[index + mX];
					if(obstacles[index - mX]) ydown  = mObstaclesVelocityY[index - mX];
					if(obstacles[index + mX*mY]) ztop    = mObstaclesVelocityZ[index + mX*mY];
					if(obstacles[index - mX*mY]) zbottom = mObstaclesVelocityZ[index - mX*mY];
					*/

					//if (obstacles[index + 1])               xright  = -vel_x[index] + mObstaclesVelocityX[index + 1];
					//if (obstacles[index - 1])               xleft   = -vel_x[index] + mObstaclesVelocityX[index - 1];
					//if (obstacles[index + dim.x()])         yup     = -vel_y[index] + mObstaclesVelocityY[index + dim.x()];
					//if (obstacles[index - dim.x()])         ydown   = -vel_y[index] + mObstaclesVelocityY[index - dim.x()];
					//if (obstacles[index + dim.x()*dim.y()]) ztop    = -vel_z[index] + mObstaclesVelocityZ[index + dim.x()*dim.y()];
					//if (obstacles[index - dim.x()*dim.y()]) zbottom = -vel_z[index] + mObstaclesVelocityZ[index - dim.x()*dim.y()];

					(*divergence)[index] = -cell_size * 0.5f *
					(
						xright - xleft +
						yup - ydown +
						ztop - zbottom
					);
				}
			}
		}
	}
	
	void CenterGrid3DPoissonSolver::buildLHS
	(
		Eigen::VectorXf& Acenter,
		Eigen::VectorXf& Ax,
		Eigen::VectorXf& Ay,
		Eigen::VectorXf& Az,
		const Eigen::VectorXi& obstacles,
		Eigen::Vector3i dim
	)
	{
		Acenter.setZero();
		Ax.setZero();
		Ay.setZero();
		Az.setZero();

		size_t index = 0;
		for (size_t z = 0; z < dim.z(); z++)
		{
			for (size_t y = 0; y < dim.y(); y++)
			{
				for (size_t x = 0; x < dim.x(); x++, index++)
				{
					if (!obstacles[index])
					{
						// Set the matrix to the Poisson stencil in order
						if (x < (dim.x() - 1) && !obstacles[index + 1])
						{
							Acenter(index) += 1.0;
							Ax(index) = -1.0;
						}
						if (x > 0 && !obstacles[index - 1])
						{
							Acenter(index) += 1.0;
						}
						if (y < (dim.y() - 1) && !obstacles[index + dim.x()])
						{
							Acenter(index) += 1.0;
							Ay(index) = -1.0;
						}
						if (y > 0 && !obstacles[index - dim.x()])
						{
							Acenter(index) += 1.0;
						}
						if (z < (dim.z() - 1) && !obstacles[index + dim.x()*dim.y()])
						{
							Acenter(index) += 1.0;
							Az(index) = -1.0;
						}
						if (z > 0 && !obstacles[index - dim.x()*dim.y()])
						{
							Acenter(index) += 1.0;
						}
					}
				}
			}
		}
	}
}}}}
#endif /* VCL_OPENMP_SUPPORT */
