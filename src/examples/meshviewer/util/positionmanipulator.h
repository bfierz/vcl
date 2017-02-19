/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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

// GSL
#include <gsl/gsl>

// VCL
#include <vcl/graphics/runtime/graphicsengine.h>

namespace Vcl { namespace Editor { namespace Util
{
	class PositionManipulator
	{
	public:
		//! Instantiate a new position handle
		PositionManipulator();

		//! Draw the handle
		void draw(
			gsl::not_null<Vcl::Graphics::Runtime::GraphicsEngine*> engine,
			const Eigen::Matrix4f& T
		);

	private:
		//! Pipeline state used for the opaque parts of the handle
		owner_ptr<Vcl::Graphics::Runtime::PipelineState> _opaquePipelineState;

		//! Pipeline state used for the transparent parts of the handle
		owner_ptr<Vcl::Graphics::Runtime::PipelineState> _transparentPipelineState;

		//! Arrow face indices
		std::unique_ptr<Vcl::Graphics::Runtime::Buffer> _indices;

		//! Arrow position buffer
		std::unique_ptr<Vcl::Graphics::Runtime::Buffer> _positions;

		//! Arrow normal buffer
		std::unique_ptr<Vcl::Graphics::Runtime::Buffer> _normals;

		//! Stride between two primitives
		uint32_t _indexStride{ 0 };

		//! Stride between two positions
		uint32_t _positionStride{ 0 };

		//! Stride between two normals
		uint32_t _normalStride{ 0 };
	};
}}}
