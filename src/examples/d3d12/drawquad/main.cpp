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
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/d3dx12.h>
#include <vcl/graphics/d3d12/commandqueue.h>
#include <vcl/graphics/d3d12/descriptortable.h>
#include <vcl/graphics/d3d12/swapchain.h>
#include <vcl/graphics/runtime/d3d12/resource/buffer.h>
#include <vcl/graphics/runtime/d3d12/resource/shader.h>
#include <vcl/graphics/runtime/d3d12/state/pipelinestate.h>

#include "quad.vs.hlsl.cso.h"
#include "quad.ps.hlsl.cso.h"

class DrawQuadApplication final : public Application
{
public:
	DrawQuadApplication()
	: Application("DrawQuads")
	{
		using Vcl::Graphics::D3D12::DescriptorTableLayout;
		using Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState;
		using Vcl::Graphics::Runtime::D3D12::Shader;
		using Vcl::Graphics::Runtime::PipelineStateDescription;
		using Vcl::Graphics::Runtime::PrimitiveType;
		using Vcl::Graphics::Runtime::RenderTargetLayout;
		using Vcl::Graphics::Runtime::ShaderType;
		using Vcl::Graphics::SurfaceFormat;

		_vs = std::make_unique<Shader>(ShaderType::VertexShader, 0, QuadCsoVS);
		_ps = std::make_unique<Shader>(ShaderType::FragmentShader, 0, QuadCsoPS);

		_tableLayout = std::make_unique<DescriptorTableLayout>(device());

		PipelineStateDescription psd;
		psd.VertexShader = _vs.get();
		psd.FragmentShader = _ps.get();
		psd.InputAssembly.Topology = PrimitiveType::Trianglelist;
		RenderTargetLayout rtd = {};
		rtd.ColourFormats = { SurfaceFormat::R8G8B8A8_UNORM };
		rtd.DepthStencilFormat = SurfaceFormat::D32_FLOAT;
		_gps = std::make_unique<GraphicsPipelineState>(device(), psd, rtd, _tableLayout.get());
	}

private:
	void renderFrame(Vcl::Graphics::Runtime::D3D12::CommandBuffer* cmd_buffer, D3D12_CPU_DESCRIPTOR_HANDLE rtv, D3D12_CPU_DESCRIPTOR_HANDLE dsv) override
	{
		using namespace Vcl::Graphics::Runtime;

		RenderPassDescription rp_desc = {};
		rp_desc.RenderTargetAttachments.resize(1);
		rp_desc.RenderTargetAttachments[0].Attachment = reinterpret_cast<void*>(rtv.ptr);
		rp_desc.RenderTargetAttachments[0].ClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		rp_desc.RenderTargetAttachments[0].LoadOp = AttachmentLoadOp::Clear;
		rp_desc.DepthStencilTargetAttachment.Attachment = reinterpret_cast<void*>(dsv.ptr);
		rp_desc.DepthStencilTargetAttachment.ClearDepth = 1.0f;
		rp_desc.DepthStencilTargetAttachment.DepthLoadOp = AttachmentLoadOp::Clear;
		cmd_buffer->beginRenderPass(rp_desc);

		const auto size = swapChain()->bufferSize();
		const auto x = size.first / 4;
		const auto y = size.second / 4;
		const auto w = size.first / 2;
		const auto h = size.second / 2;

		D3D12_VIEWPORT vp{ x, y, w, h, 0, 1 };
		cmd_buffer->handle()->RSSetViewports(1, &vp);
		D3D12_RECT sr{ x, y, x+w, y+h };
		cmd_buffer->handle()->RSSetScissorRects(1, &sr);

		cmd_buffer->bindPipeline(_gps.get());
		cmd_buffer->handle()->SetGraphicsRootSignature(_tableLayout->rootSignature());
		cmd_buffer->handle()->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		cmd_buffer->draw(6, 1, 0, 0);

		cmd_buffer->endRenderPass();
	}

	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Shader> _vs;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::Shader> _ps;

	std::unique_ptr<Vcl::Graphics::D3D12::DescriptorTableLayout> _tableLayout;
	std::unique_ptr<Vcl::Graphics::Runtime::D3D12::GraphicsPipelineState> _gps;
};

int main(int argc, char** argv)
{
	DrawQuadApplication app;
	return app.run();
}
