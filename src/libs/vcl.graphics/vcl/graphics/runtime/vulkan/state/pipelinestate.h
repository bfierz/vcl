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
#include <vcl/graphics/vulkan/descriptorheap.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {
	class GraphicsPipelineState : public Runtime::PipelineState
	{
	public:
		GraphicsPipelineState(
			Vcl::Graphics::Vulkan::Context* context,
			const Vcl::Graphics::Vulkan::DescriptorHeap* heap,
			const PipelineStateDescription& desc,
			const RenderTargetLayout& rt_layout);
		~GraphicsPipelineState();

		//! Convert to Vulkan ID
		inline operator VkPipeline() const { return _state; }

		//! Access the bindless pipeline layout shared by all pipelines
		VkPipelineLayout layout() const { return _layout; }

		//! Access the input layout
		const InputLayoutDescription& inputLayout() const { return _inputLayout; }

	private:
		//! Owner
		Vcl::Graphics::Vulkan::Context* _context;

		//! Layout of the descriptor heap the pipeline was created for
		VkPipelineLayout _layout;

		//! Input layout
		InputLayoutDescription _inputLayout;

		//! Vulkan pipeline state
		VkPipeline _state{ VK_NULL_HANDLE };
	};

	//! Pipeline generating its primitives in a mesh shader (VK_EXT_mesh_shader).
	//! Requires Context::supportsMeshShaders().
	class MeshPipelineState : public Runtime::PipelineState
	{
	public:
		MeshPipelineState(
			Vcl::Graphics::Vulkan::Context* context,
			const Vcl::Graphics::Vulkan::DescriptorHeap* heap,
			const MeshPipelineStateDescription& desc,
			const RenderTargetLayout& rt_layout);
		~MeshPipelineState();

		//! Convert to Vulkan ID
		inline operator VkPipeline() const { return _state; }

		//! Access the bindless pipeline layout shared by all pipelines
		VkPipelineLayout layout() const { return _layout; }

	private:
		//! Owner
		Vcl::Graphics::Vulkan::Context* _context;

		//! Layout of the descriptor heap the pipeline was created for
		VkPipelineLayout _layout;

		//! Vulkan pipeline state
		VkPipeline _state{ VK_NULL_HANDLE };
	};

	class ComputePipelineState : public Runtime::PipelineState
	{
	public:
		ComputePipelineState(Vcl::Graphics::Vulkan::Context* context, const Vcl::Graphics::Vulkan::DescriptorHeap* heap, const ComputePipelineStateDescription& desc);
		~ComputePipelineState();

		//! Convert to Vulkan ID
		inline operator VkPipeline() const { return _state; }

		//! Access the bindless pipeline layout shared by all pipelines
		VkPipelineLayout layout() const { return _layout; }

	private:
		//! Owner
		Vcl::Graphics::Vulkan::Context* _context;

		//! Layout of the descriptor heap the pipeline was created for
		VkPipelineLayout _layout;

		//! Vulkan pipeline state
		VkPipeline _state{ VK_NULL_HANDLE };
	};
}}}}
