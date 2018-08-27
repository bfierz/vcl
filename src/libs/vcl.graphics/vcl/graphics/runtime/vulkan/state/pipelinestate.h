/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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

// C++ standard libary
#include <memory>

// VCL
#include <vcl/core/flags.h>
#include <vcl/core/span.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/descriptorset.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan
{
	class PipelineLayout
	{
		using DescriptorSetLayout = Graphics::Vulkan::DescriptorSetLayout;
		using PushConstantDescriptor = Graphics::Vulkan::PushConstantDescriptor;
	public:
		//! Constructor
		PipelineLayout() = default;

		//! Constructor
		PipelineLayout(Vcl::Graphics::Vulkan::Context* context,
			           const DescriptorSetLayout* descriptor_set_layout,
			           stdext::span<const PushConstantDescriptor> push_constants);

		//! Copy Constructor
		PipelineLayout(const PipelineLayout&) = delete;

		//! Move Constructor
		PipelineLayout(PipelineLayout&&);

		//! Destructor
		~PipelineLayout();

		//! Convert to Vulkan ID
		inline operator VkPipelineLayout() const { return _layout; }

	private:
		//! Link to the context owning the layout
		Vcl::Graphics::Vulkan::Context* _context{ nullptr };

		//! Vulkan pipeline layout
		VkPipelineLayout _layout{ nullptr };

		//! Associated descriptor set layout
		const DescriptorSetLayout* _descriptorSetLayout{ nullptr };
	};

	class ComputePipelineState : public Runtime::PipelineState
	{
	public:
		ComputePipelineState(Vcl::Graphics::Vulkan::Context* context, const PipelineLayout* layout, const ComputePipelineStateDescription& desc);
		~ComputePipelineState();

		//! Convert to Vulkan ID
		inline operator VkPipeline() const { return _state; }

		//! Access the pipeline layout
		const PipelineLayout& layout() const { return *_layout; }

	private:
		//! Owner
		Vcl::Graphics::Vulkan::Context* _context;

		//! Associated layout
		const PipelineLayout* _layout;

		//! Vulkan pipeline state
		VkPipeline _state;
	};
}}}}
