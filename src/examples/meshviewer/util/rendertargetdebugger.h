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

// VCL
#include <vcl/graphics/imageprocessing/imageprocessor.h>
#include <vcl/graphics/runtime/state/sampler.h>
#include <vcl/graphics/runtime/graphicsengine.h>

namespace Vcl { namespace Editor { namespace Util
{
	class RendertargetDebugger
	{
	public:
		//! Instantiate a new debug renderer
		RendertargetDebugger();

		//! Draw a texture to the current render target
		//! \param engine Graphics engine to use
		//! \param texture View on the texture to draw
		//! \param max_value Maximum integer value in the texture
		//! \param loc_size Location and size of output region
		void draw(
			Vcl::ref_ptr<Vcl::Graphics::Runtime::GraphicsEngine> engine,
			const Vcl::Graphics::Runtime::Texture& texture,
			const unsigned int max_value,
			const Eigen::Vector4f& loc_size
		);

	private:
		//! Image processor used to display the render-target
		owner_ptr<Vcl::Graphics::ImageProcessing::ImageProcessor> _imageProcessor;

		//! Image processing graph for 32-bit integers
		owner_ptr<Vcl::Graphics::ImageProcessing::Task> _integerTaskGraph;

		//! Pipeline state to present the image to the screen
		owner_ptr<Vcl::Graphics::Runtime::PipelineState> _presentationPipelineState;

		//! Sampler for displaying the render-target
		std::unique_ptr<Vcl::Graphics::Runtime::Sampler> _rtSampler;
	};
}}}
