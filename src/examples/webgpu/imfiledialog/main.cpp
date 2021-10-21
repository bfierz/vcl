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

#include <unordered_map>

#include <vcl/graphics/webgpu/helpers.h>

#include "../common/imguiapp.h"

#include "ImFileDialog.h"

static WGPUBuffer ImGui_ImplWGPU_CreateBufferFromData(const WGPUDevice& device, const void* data, uint64_t size, WGPUBufferUsage usage)
{
	WGPUBufferDescriptor descriptor = {};
	descriptor.size = size;
	descriptor.usage = usage | WGPUBufferUsage_CopyDst;
	WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &descriptor);

	WGPUQueue queue = wgpuDeviceGetQueue(device);
	wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
	return buffer;
}

class DemoImGuiApplication final : public ImGuiApplication
{
public:
	DemoImGuiApplication(const char* title)
	: ImGuiApplication(title)
	{
		// ImFileDialog requires you to set the CreateTexture and DeleteTexture
		ifd::FileDialog::Instance().CreateTexture = [this](uint8_t* data, int w, int h, char fmt) -> void* {
			using Vcl::Graphics::WebGPU::createTextureFromData;
			const auto texture_data = createTextureFromData(_wgpuDevice, Vcl::Graphics::SurfaceFormat::R8G8B8A8_UNORM, w, h, stdext::make_span(data, 4 * w * h));
			fi_img_cache.emplace(std::get<1>(texture_data), std::get<0>(texture_data));
			return (void*)std::get<1>(texture_data);
		};
		ifd::FileDialog::Instance().DeleteTexture = [this](void* tex) {
			WGPUTextureView tex_view = (WGPUTextureView)((uintptr_t)tex);
			auto entry = fi_img_cache.find(tex_view);
			wgpuTextureRelease(entry->second);
			wgpuTextureViewRelease(tex_view);
			fi_img_cache.erase(entry);
		};
	}

private:
	void updateFrame() override
	{
		ImGuiApplication::updateFrame();

		ImGui::Begin("Control Panel");
		if (ImGui::Button("Open file"))
			ifd::FileDialog::Instance().Open("ShaderOpenDialog", "Open a shader", "Image file (*.png;*.jpg;*.jpeg;*.bmp;*.tga){.png,.jpg,.jpeg,.bmp,.tga},.*", true);
		if (ImGui::Button("Open directory"))
			ifd::FileDialog::Instance().Open("DirectoryOpenDialog", "Open a directory", "");
		if (ImGui::Button("Save file"))
			ifd::FileDialog::Instance().Save("ShaderSaveDialog", "Save a shader", "*.sprj {.sprj}");
		ImGui::End();

		if (ifd::FileDialog::Instance().IsDone("ShaderOpenDialog"))
		{
			if (ifd::FileDialog::Instance().HasResult())
			{
				const std::vector<std::filesystem::path>& res = ifd::FileDialog::Instance().GetResults();
				for (const auto& r : res) // ShaderOpenDialog supports multiselection
					printf("OPEN[%s]\n", r.u8string().c_str());
			}
			ifd::FileDialog::Instance().Close();
		}
		if (ifd::FileDialog::Instance().IsDone("DirectoryOpenDialog"))
		{
			if (ifd::FileDialog::Instance().HasResult())
			{
				std::string res = ifd::FileDialog::Instance().GetResult().u8string();
				printf("DIRECTORY[%s]\n", res.c_str());
			}
			ifd::FileDialog::Instance().Close();
		}
		if (ifd::FileDialog::Instance().IsDone("ShaderSaveDialog"))
		{
			if (ifd::FileDialog::Instance().HasResult())
			{
				std::string res = ifd::FileDialog::Instance().GetResult().u8string();
				printf("SAVE[%s]\n", res.c_str());
			}
			ifd::FileDialog::Instance().Close();
		}
	}
	void renderFrame(WGPUTextureView back_buffer) override
	{
		std::array<WGPURenderPassColorAttachment, 1> color_attachments = {};
		color_attachments[0].loadOp = WGPULoadOp_Clear;
		color_attachments[0].storeOp = WGPUStoreOp_Store;
		color_attachments[0].clearColor = { clear_color.x, clear_color.y, clear_color.z, clear_color.w };
		color_attachments[0].view = back_buffer;
		WGPURenderPassDescriptor render_pass_desc = {};
		render_pass_desc.colorAttachmentCount = static_cast<uint32_t>(color_attachments.size());
		render_pass_desc.colorAttachments = color_attachments.data();
		render_pass_desc.depthStencilAttachment = nullptr;

		WGPUCommandEncoderDescriptor enc_desc = {};
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_wgpuDevice, &enc_desc);
		WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);
		wgpuRenderPassEncoderEndPass(pass);

		WGPUCommandBufferDescriptor cmd_buffer_desc = {};
		WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);
		WGPUQueue queue = wgpuDeviceGetQueue(_wgpuDevice);
		wgpuQueueSubmit(queue, 1, &cmd_buffer);

		ImGuiApplication::renderFrame(back_buffer);
	}

	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	std::unordered_map<WGPUTextureView, WGPUTexture> fi_img_cache;
};

int main(int argc, char** argv)
{
	DemoImGuiApplication app{ "ImFileDialog Demo" };
	return app.run();
}
