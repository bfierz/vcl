/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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

// C++ standard library
#include <string>

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/descriptorset.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	class PipelineLayout final
	{
	public:
		//! Constructor
		PipelineLayout(Context* context, DescriptorSetLayout* descriptor_set_layout);

		//! Destructor
		~PipelineLayout();

		//! Convert to Vulkan ID
		inline operator VkPipelineLayout() const
		{
			return _layout;
		}

	public:

	private:
		//! Owner
		Context* _context;

		//! Vulkan pipeline layout
		VkPipelineLayout _layout;

		//! Associated descriptor set layout
		DescriptorSetLayout* _descriptorSetLayout;
	};

	class PipelineState final : public Vcl::Graphics::Runtime::PipelineState
	{
	public:
		//! Constructor
		PipelineState
		(
			Context* context, PipelineLayout* layout, VkRenderPass pass,
			const Vcl::Graphics::Runtime::PipelineStateDescription& desc
		);

		//! Destructor
		~PipelineState();

		//! Convert to Vulkan ID
		inline operator VkPipeline() const
		{
			return _state;
		}

	public:
		
	private:
		//! Owner
		Context* _context;

		//! Associated layout
		PipelineLayout* _layout;

		//! Vulkan pipeline state
		VkPipeline _state;
	};
}}}
