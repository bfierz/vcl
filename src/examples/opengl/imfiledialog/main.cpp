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

#include "../common/imguiapp.h"

#include "ImFileDialog.h"

// VCL
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>

class DemoImGuiApplication final : public ImGuiApplication
{
public:
	using Texture2D = Vcl::Graphics::Runtime::OpenGL::Texture2D;

	DemoImGuiApplication(const char* title)
	: ImGuiApplication(title)
	{
		// ImFileDialog requires you to set the CreateTexture and DeleteTexture
		ifd::FileDialog::Instance().CreateTexture = [this](uint8_t* data, int w, int h, char fmt) -> void* {
			/*Vcl::Graphics::Runtime::Texture2DDescription desc;
			//desc.Format = (fmt == 0) ? Vcl::Graphics::SurfaceFormat::B8G8R8A8_UNORM : Vcl::Graphics::SurfaceFormat::R8G8B8A8_UNORM;
			desc.Format = Vcl::Graphics::SurfaceFormat::R8G8B8A8_UNORM;
			desc.Usage = Vcl::Graphics::Runtime::TextureUsage::Sampled;
			desc.Width = w;
			desc.Height = h;
			desc.MipLevels = 1;
			desc.ArraySize = 1;
			
			Texture2D tex{desc};
			GLuint tex_id = tex.id();
			fi_img_cache.emplace(tex_id, std::move(tex));

			return (void*)tex_id;*/

			GLuint tex;
			glGenTextures(1, &tex);
			glBindTexture(GL_TEXTURE_2D, tex);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, (fmt == 0) ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, 0);
			return (void*)tex;
		};
		ifd::FileDialog::Instance().DeleteTexture = [this](void* tex) {
			GLuint tex_id = (GLuint)((uintptr_t)tex);
			//fi_img_cache.erase(fi_img_cache.find(tex_id));
			glDeleteTextures(1, &tex_id);
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
	void renderFrame() override
	{
		glClearBufferfv(GL_COLOR, 0, (float*)&clear_color);
		glClearBufferfi(GL_DEPTH_STENCIL, 0, 1.0f, 0);

		ImGuiApplication::renderFrame();
	}

	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	std::unordered_map<GLuint, Texture2D> fi_img_cache;
};

int main(int argc, char** argv)
{
	DemoImGuiApplication app{ "ImFileDialog Demo" };
	return app.run();
}
