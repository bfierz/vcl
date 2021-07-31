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

#include "../common/app.h"

// Include the relevant parts from the library
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/runtime/webgpu/resource/buffer.h>
#include <vcl/graphics/runtime/webgpu/resource/shader.h>
#include <vcl/graphics/runtime/webgpu/state/pipelinestate.h>
#include <vcl/graphics/surfaceformat.h>

#include "quad.vert.spv.h"
#include "quad.frag.spv.h"

class DrawQuadApplication final : public Application
{
public:
	DrawQuadApplication()
	: Application("DrawQuads")
	{
		using Vcl::Graphics::SurfaceFormat;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::PrimitiveType;
		using Vcl::Graphics::Runtime::RenderTargetLayout;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::Runtime::WebGPU::GraphicsPipelineState;
		using Vcl::Graphics::Runtime::WebGPU::Shader;

		_vs = std::make_unique<Shader>(_wgpuDevice, ShaderType::VertexShader, 0, QuadSpirvVS);
		_fs = std::make_unique<Shader>(_wgpuDevice, ShaderType::FragmentShader, 0, QuadSpirvFS);

		PipelineStateDescription psd;
		psd.VertexShader = _vs.get();
		psd.FragmentShader = _fs.get();
		psd.InputAssembly.Topology = PrimitiveType::Trianglelist;
		RenderTargetLayout rtd = {};
		rtd.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
		rtd.DepthStencilFormat = SurfaceFormat::Unknown;
		_gps = std::make_unique<GraphicsPipelineState>(_wgpuDevice, psd, rtd);
	}

private:
	void renderFrame(WGPUTextureView back_buffer) override
	{
		const auto size = _swapChainSize;
		const auto x = size.first / 4;
		const auto y = size.second / 4;
		const auto w = size.first / 2;
		const auto h = size.second / 2;

		std::array<WGPURenderPassColorAttachmentDescriptor, 1> color_attachments = {};
		color_attachments[0].loadOp = WGPULoadOp_Clear;
		color_attachments[0].storeOp = WGPUStoreOp_Store;
		color_attachments[0].clearColor = { 1.0f, 0.0f, 1.0f, 0.0f };
		color_attachments[0].view = back_buffer;
		WGPURenderPassDepthStencilAttachmentDescriptor depth_attachment = {};
		depth_attachment.clearDepth = 1.0f;
		depth_attachment.clearStencil = 0;
		depth_attachment.depthLoadOp = WGPULoadOp_Clear;
		depth_attachment.depthStoreOp = WGPUStoreOp_Store;
		depth_attachment.stencilLoadOp = WGPULoadOp_Clear;
		depth_attachment.stencilStoreOp = WGPUStoreOp_Store;
		WGPURenderPassDescriptor render_pass_desc = {};
		render_pass_desc.colorAttachmentCount = static_cast<uint32_t>(color_attachments.size());
		render_pass_desc.colorAttachments = color_attachments.data();
		render_pass_desc.depthStencilAttachment = nullptr;

		WGPUCommandEncoderDescriptor enc_desc = {};
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_wgpuDevice, &enc_desc);
		{
			WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);
			wgpuRenderPassEncoderSetViewport(pass, x, y, w, h, 0, 1);
			wgpuRenderPassEncoderSetScissorRect(pass, x, y, w, h);
			wgpuRenderPassEncoderSetPipeline(pass, _gps->handle());
			wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
			wgpuRenderPassEncoderEndPass(pass);
			wgpuRenderPassEncoderRelease(pass);
		}

		WGPUCommandBufferDescriptor cmd_buffer_desc = {};
		WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);
		wgpuCommandEncoderRelease(encoder);
		WGPUQueue queue = wgpuDeviceGetQueue(_wgpuDevice);
		wgpuQueueSubmit(queue, 1, &cmd_buffer);
		wgpuCommandBufferRelease(cmd_buffer);
	}

	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::Shader> _vs;
	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::Shader> _fs;

	std::unique_ptr<Vcl::Graphics::Runtime::WebGPU::GraphicsPipelineState> _gps;
};

// Declare application as global object instead of stack object in main
// in order to prevent it to be cleaned up,
// when 'emscripten_set_main_loop' exists.
DrawQuadApplication app;

int main(int argc, char** argv)
{
	return app.run();
}
