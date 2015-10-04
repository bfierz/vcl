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
#include <set>
#include <unordered_map>
#include <vector>

// VCL
#include <vcl/core/handle.h>
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/kernel.h>
#include <vcl/compute/module.h>
#include <vcl/physics/fluid/centergrid.h>

namespace Vcl { namespace Physics { namespace Fluid { namespace Cuda
{
	enum class BorderOp
	{
		SetZero,
		Copy,
		Neumann
	};

	class CenterGrid : public Fluid::CenterGrid
	{
	public:
		template<typename T> using ref_ptr = Vcl::Core::ref_ptr<T>;

	public:
		CenterGrid(ref_ptr<Vcl::Compute::Cuda::Context> ctx, ref_ptr<Vcl::Compute::Cuda::CommandQueue> queue, const Eigen::Vector3i& resolution, float spacing = -1);
		virtual ~CenterGrid();

	public:
		virtual void resize(const Eigen::Vector3i& resolution) override;

		virtual void swap() override;

	public:
		bool hasNamedBuffer(const std::string& name) const;
		ref_ptr<Compute::Buffer> namedBuffer(const std::string& name) const;
		ref_ptr<Compute::Buffer> addNamedBuffer(const std::string& name);

	public:

		void setBorderZero(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;
		void setBorderZeroX(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;
		void setBorderZeroY(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;
		void setBorderZeroZ(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;

		void copyBorderX(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;
		void copyBorderY(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;
		void copyBorderZ(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;

		void setNeumannX(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;
		void setNeumannY(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;
		void setNeumannZ(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;

		void accumulate(Vcl::Compute::Cuda::CommandQueue& queue, Compute::Cuda::Buffer& dst, const Compute::Cuda::Buffer& src, float alpha) const;

	private:

		void setBorderX(Vcl::Compute::Cuda::CommandQueue& queue, BorderOp mode, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;
		void setBorderY(Vcl::Compute::Cuda::CommandQueue& queue, BorderOp mode, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;
		void setBorderZ(Vcl::Compute::Cuda::CommandQueue& queue, BorderOp mode, Compute::Cuda::Buffer& buffer, Eigen::Vector3i dim) const;

	public:
		Compute::Cuda::Buffer& forces(int dim);
		Compute::Cuda::Buffer& velocities(int i, int dim);
		const Compute::Cuda::Buffer& velocities(int i, int dim) const;

		ref_ptr<Compute::Buffer> obstacles() const;
		Compute::Cuda::Buffer&  obstacleBuffer();

		ref_ptr<Compute::Buffer> densities(int i) const;
		Compute::Cuda::Buffer&  densityBuffer(int i);

		ref_ptr<Compute::Buffer> heat(int i) const;
		Compute::Cuda::Buffer&  heatBuffer(int i);

		Compute::Cuda::Buffer& vorticityMag();
		Compute::Cuda::Buffer& vorticity(int dim);

	public:
		ref_ptr<Compute::Buffer> aquireIntermediateBuffer();
		void releaseIntermediateBuffer(ref_ptr<Compute::Buffer> handle);

	private:
		ref_ptr<Vcl::Compute::Cuda::Context> _ownerCtx;

		std::array<ref_ptr<Compute::Buffer>, 3> _force;

		std::array<std::array<ref_ptr<Compute::Buffer>, 3>, 2> _velocity;
		
		ref_ptr<Compute::Buffer>                _vorticityMag;
		std::array<ref_ptr<Compute::Buffer>, 3> _vorticity;

		ref_ptr<Compute::Buffer>                _obstacles;
		std::array<ref_ptr<Compute::Buffer>, 3> _obstaclesVelocity;

		std::array<ref_ptr<Compute::Buffer>, 2> _density;
		
		std::array<ref_ptr<Compute::Buffer>, 2> _heat;

	private:
		std::unordered_map<std::string, ref_ptr<Compute::Buffer>> _namedBuffers;

	private: // Intermediate buffer management
		std::vector<ref_ptr<Compute::Buffer>> _freeTmpBuffers;
		std::set<ref_ptr<Compute::Buffer>> _aquiredTmpBuffers;

	private: // Kernel functions

		//! Module with the cuda fluid code
		ref_ptr< Compute::Module> _fluidModule;

		//! Device function ensuring correct boundaries in x direction
		ref_ptr<Vcl::Compute::Cuda::Kernel> _setBorderX = nullptr;

		//! Device function ensuring correct boundaries in y direction
		ref_ptr<Vcl::Compute::Cuda::Kernel> _setBorderY = nullptr;

		//! Device function ensuring correct boundaries in z direction
		ref_ptr<Vcl::Compute::Cuda::Kernel> _setBorderZ = nullptr;

		//! Device function computing a saxpy
		ref_ptr<Vcl::Compute::Cuda::Kernel> _accumulate = nullptr;
	};
}}}}
