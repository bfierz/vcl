/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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

// VCL configuration
#include <vcl/config/global.h>

// C++ Standard Library
#include <vector>

// Include the relevant parts from the library
#include <vcl/graphics/imageprocessing/opengl/luminance.h>
#include <vcl/graphics/imageprocessing/opengl/srgb.h>
#include <vcl/graphics/imageprocessing/opengl/tonemap.h>
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>

// Google test
#include <gtest/gtest.h>

TEST(OpenGL, ImageProcessingTaskSRGB)
{
	using namespace Vcl::Graphics::ImageProcessing::OpenGL;
	using namespace Vcl::Graphics;

	// Instantiate the image processor
	ImageProcessor proc;

	// Prepare the input
	std::vector<unsigned char> numerical_average_gray(256 * 256 * 4, 127);
	std::vector<unsigned char> physiological_average_gray(256 * 256 * 4, 127);

	Runtime::TextureResource init_res;
	init_res.Width = 256;
	init_res.Height = 256;
	init_res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_res.Data = numerical_average_gray.data();

	Runtime::Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 1;
	desc2d.Width = 256;
	desc2d.Height = 256;
	desc2d.MipLevels = 1;
	Runtime::OpenGL::Texture2D input{ desc2d, &init_res };

	// Task to test
	SRGB task{ &proc };

	// Configure the input
	task.inputSlot(0)->setResource(&input);

	// Memory barrier ensuring safe writes
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Execute the kernel
	proc.execute(&task);

	// Memory barrier ensuring safe reads
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Read the output
	auto output = task.outputSlot(0)->resource();
	auto& out_tex = static_cast<const Runtime::OpenGL::Texture2D&>(*output);
	out_tex.read(physiological_average_gray.size(), physiological_average_gray.data());

	bool equal = true;
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			bool r = physiological_average_gray[4 * (y * 256 + x) + 0] == 187;
			bool g = physiological_average_gray[4 * (y * 256 + x) + 1] == 187;
			bool b = physiological_average_gray[4 * (y * 256 + x) + 2] == 187;

			equal = equal && r && g && b;
		}
	}

	EXPECT_TRUE(equal) << "SRGB transform was not successful";

	return;
}

TEST(OpenGL, ImageProcessingTaskLuminance)
{
	using namespace Vcl::Graphics::ImageProcessing::OpenGL;
	using namespace Vcl::Graphics;

	// Instantiate the image processor
	ImageProcessor proc;

	// Prepare the input
	std::vector<unsigned char> input_pattern(512 * 128 * 4, 127);
	std::vector<unsigned char> luminance(1 * 1 * 2, 0);

	Runtime::TextureResource init_res;
	init_res.Width = 512;
	init_res.Height = 128;
	init_res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_res.Data = input_pattern.data();

	Runtime::Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 1;
	desc2d.Width = 512;
	desc2d.Height = 128;
	desc2d.MipLevels = 1;
	Runtime::OpenGL::Texture2D input{ desc2d, &init_res };

	// Task to test
	Luminance task{ &proc };

	// Configure the input
	task.inputSlot(0)->setResource(&input);

	// Memory barrier ensuring safe writes
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Execute the kernel
	proc.execute(&task);

	// Memory barrier ensuring safe reads
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Read the output
	auto output = task.outputSlot(0)->resource();
	auto& out_tex = static_cast<const Runtime::OpenGL::Texture2D&>(*output);

	glPixelStorei(GL_PACK_ALIGNMENT, 2);
	out_tex.read(luminance.size(), luminance.data());

	return;
}

TEST(OpenGL, ImageProcessingTaskTonemap)
{
	using namespace Vcl::Graphics::ImageProcessing::OpenGL;
	using namespace Vcl::Graphics;

	// Instantiate the image processor
	ImageProcessor proc;

	// Prepare the input
	std::vector<unsigned char> numerical_average_gray(256 * 256 * 4, 127);
	std::vector<unsigned char> physiological_average_gray(256 * 256 * 4, 127);

	float lum = 0.5f;

	Runtime::TextureResource init_res;
	init_res.Width = 256;
	init_res.Height = 256;
	init_res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_res.Data = numerical_average_gray.data();

	Runtime::Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 1;
	desc2d.Width = 256;
	desc2d.Height = 256;
	desc2d.MipLevels = 1;
	Runtime::OpenGL::Texture2D input{ desc2d, &init_res };

	Runtime::TextureResource init_avg_lum;
	init_avg_lum.Width = 1;
	init_avg_lum.Height = 1;
	init_avg_lum.Format = SurfaceFormat::R32_FLOAT;
	init_avg_lum.Data = &lum;

	Runtime::Texture2DDescription desc_avg_lum;
	desc_avg_lum.Format = SurfaceFormat::R16_FLOAT;
	desc_avg_lum.ArraySize = 1;
	desc_avg_lum.Width = 1;
	desc_avg_lum.Height = 1;
	desc_avg_lum.MipLevels = 1;
	Runtime::OpenGL::Texture2D avg_lum{ desc_avg_lum, &init_avg_lum };

	// Task to test
	Tonemap task{ &proc };

	// Configure the input
	task.inputSlot(0)->setResource(&input);
	task.inputSlot(1)->setResource(&avg_lum);

	// Memory barrier ensuring safe writes
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Execute the kernel
	proc.execute(&task);

	// Memory barrier ensuring safe reads
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Read the output
	auto output = task.outputSlot(0)->resource();
	auto& out_tex = static_cast<const Runtime::OpenGL::Texture2D&>(*output);
	out_tex.read(physiological_average_gray.size(), physiological_average_gray.data());

	bool equal = true;
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			bool r = physiological_average_gray[4 * (y * 256 + x) + 0] == 187;
			bool g = physiological_average_gray[4 * (y * 256 + x) + 1] == 187;
			bool b = physiological_average_gray[4 * (y * 256 + x) + 2] == 187;

			equal = equal && r && g && b;
		}
	}

	EXPECT_TRUE(equal) << "Tonemapping was not successful";

	return;
}
