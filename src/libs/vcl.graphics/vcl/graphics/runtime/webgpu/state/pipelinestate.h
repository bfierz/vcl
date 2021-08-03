/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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
#include <vcl/config/webgpu.h>

// VCL
#include <vcl/core/span.h>
#include <vcl/graphics/runtime/state/pipelinestate.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace WebGPU {
	class GraphicsPipelineState : public Runtime::PipelineState
	{
	public:
		GraphicsPipelineState(
			WGPUDevice device,
			const PipelineStateDescription& desc,
			const RenderTargetLayout& rt_formats);

		//! WebGPU handle
		WGPURenderPipeline handle() const { return _pipeline; }

		const InputLayoutDescription& inputLayout() const { return _inputLayout; }

	private:
		WGPURenderPipeline _pipeline;

		InputLayoutDescription _inputLayout;
	};

	class ComputePipelineState : public Runtime::PipelineState
	{
	public:
		ComputePipelineState(
			WGPUDevice device,
			const ComputePipelineStateDescription& desc);

		//! WebGPU handle
		WGPUComputePipeline handle() const { return _pipeline; }

	private:
		WGPUComputePipeline _pipeline;
	};
}}}}
