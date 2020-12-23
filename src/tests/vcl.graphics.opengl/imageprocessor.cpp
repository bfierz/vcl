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
#include <vcl/graphics/imageprocessing/opengl/gaussian.h>
#include <vcl/graphics/imageprocessing/opengl/luminance.h>
#include <vcl/graphics/imageprocessing/opengl/srgb.h>
#include <vcl/graphics/imageprocessing/opengl/tonemap.h>
#include <vcl/graphics/runtime/opengl/resource/texture2d.h>

// Google test
#include <gtest/gtest.h>

// Support code
#include "bitmap.h"

// Sample images
#include "pattern.h"

// http://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
// https://gist.github.com/rygorous/2156668

union FP32
{
	uint32_t u;
	float f;
	struct
	{
		uint32_t Mantissa : 23;
		uint32_t Exponent : 8;
		uint32_t Sign : 1;
	};
};

union FP16
{
	unsigned short u;
	struct
	{
		uint32_t Mantissa : 10;
		uint32_t Exponent : 5;
		uint32_t Sign : 1;
	};
};

// half_to_floast_fast5: slightly different approach, turns FP16 denormals into FP32 denormals.
// it's very slick and short but will be slower if denormals actually occur.
static FP32 half_to_float_fast5(FP16 h)
{
	static const FP32 magic = { (254 - 15) << 23 };
	static const FP32 was_infnan = { (127 + 16) << 23 };
	FP32 o;

	o.u = (h.u & 0x7fff) << 13; // exponent/mantissa bits
	o.f *= magic.f; // exponent adjust
	if (o.f >= was_infnan.f) // make sure Inf/NaN survive
		o.u |= 255 << 23;
	o.u |= (h.u & 0x8000) << 16; // sign bit
	return o;
}

// Approximate solution. This is faster but converts some sNaNs to
// infinity and doesn't round correctly. Handle with care.
static FP16 approx_float_to_half(FP32 f)
{
	FP32 f32infty = { 255 << 23 };
	FP32 f16max = { (127 + 16) << 23 };
	FP32 magic = { 15 << 23 };
	FP32 expinf = { (255 ^ 31) << 23 };
	uint32_t sign_mask = 0x80000000u;
	FP16 o = { 0 };

	uint32_t sign = f.u & sign_mask;
	f.u ^= sign;

	if (!(f.f < f32infty.u)) // Inf or NaN
		o.u = f.u ^ expinf.u;
	else
	{
		if (f.f > f16max.f) f.f = f16max.f;
		f.f *= magic.f;
	}

	o.u = f.u >> 13; // Take the mantissa bits
	o.u |= sign >> 16;
	return o;
}


TEST(OpenGL, ImageProcessingTaskSRGB)
{
	using namespace Vcl::Graphics::ImageProcessing::OpenGL;
	using namespace Vcl::Graphics;

	// Instantiate the image processor
	ImageProcessor proc;

	// Prepare the input
	FP32 half;
	half.f = 0.5f;
	std::vector<FP16> numerical_average_gray_half(256 * 256 * 4, approx_float_to_half(half));
	std::vector<unsigned char> numerical_average_gray(256 * 256 * 4, 128);
	std::vector<unsigned char> physiological_average_gray(256 * 256 * 4, 128);

	Runtime::TextureResource init_res;
	init_res.Width = 256;
	init_res.Height = 256;
	init_res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_res.Data = stdext::make_span(numerical_average_gray);

	// Using half-floats to load the textures causes read-bug in the shader. G and A are read as 0.
	//init_res.Format = SurfaceFormat::R16G16B16A16_FLOAT;
	//init_res.Data = stdext::make_span(reinterpret_cast<const uint8_t*>(numerical_average_gray_half.data()), numerical_average_gray_half.size() * sizeof(decltype(numerical_average_gray_half)::value_type));

	Runtime::Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R16G16B16A16_FLOAT;
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
			bool r = physiological_average_gray[4 * (y * 256 + x) + 0] == 188;
			bool g = physiological_average_gray[4 * (y * 256 + x) + 1] == 188;
			bool b = physiological_average_gray[4 * (y * 256 + x) + 2] == 188;
			bool a = physiological_average_gray[4 * (y * 256 + x) + 3] == 128;

			equal = equal && r && g && b;
		}
	}

	EXPECT_TRUE(equal) << "SRGB transform was not successful";
}

TEST(OpenGL, ImageProcessingTaskLuminance)
{
	using namespace Vcl::Graphics::ImageProcessing::OpenGL;
	using namespace Vcl::Graphics;

	// Instantiate the image processor
	ImageProcessor proc;

	// Prepare the input
	std::vector<unsigned char> input_pattern(512 * 128 * 4, 127);
	std::vector<FP16> luminance(1 * 1);

	Runtime::TextureResource init_res;
	init_res.Width = 512;
	init_res.Height = 128;
	init_res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_res.Data = input_pattern;

	Runtime::Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R16G16B16A16_FLOAT;
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
	out_tex.read(luminance.size() * sizeof(FP16), luminance.data());

	float log_lum = half_to_float_fast5(luminance[0]).f;
	float scaled_lum = roundf(255.0f * std::exp(log_lum));
	unsigned char lum = (unsigned char)scaled_lum;

	EXPECT_EQ(127, lum) << "Luminance computation was not successful";
}

TEST(OpenGL, ImageProcessingTaskTonemap)
{
	using namespace Vcl::Graphics::ImageProcessing::OpenGL;
	using namespace Vcl::Graphics;

	// Instantiate the image processor
	ImageProcessor proc;

	// Prepare the input
	std::vector<std::array<unsigned char, 4>> numerical_average_gray(256 * 256);
	std::vector<std::array<unsigned char, 4>> physiological_average_gray(256 * 256);

	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			numerical_average_gray[y * 256 + x] = { 127, 127, 127, 127 };
		}
	}

	float lum = std::log(0.5f);

	Runtime::TextureResource init_res;
	init_res.Width = 256;
	init_res.Height = 256;
	init_res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_res.Data = stdext::make_span(reinterpret_cast<const uint8_t*>(numerical_average_gray.data()), numerical_average_gray.size() * sizeof(decltype(numerical_average_gray)::value_type));

	Runtime::Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R16G16B16A16_FLOAT;
	desc2d.ArraySize = 1;
	desc2d.Width = 256;
	desc2d.Height = 256;
	desc2d.MipLevels = 1;
	Runtime::OpenGL::Texture2D input{ desc2d, &init_res };

	Runtime::TextureResource init_avg_lum;
	init_avg_lum.Width = 1;
	init_avg_lum.Height = 1;
	init_avg_lum.Format = SurfaceFormat::R32_FLOAT;
	init_avg_lum.Data = stdext::make_span(reinterpret_cast<const uint8_t*>(&lum), sizeof(float));

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
	out_tex.read(physiological_average_gray.size() * 4, physiological_average_gray.data());

	bool equal = true;
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			bool r = physiological_average_gray[y * 256 + x][0] == 101;
			bool g = physiological_average_gray[y * 256 + x][1] == 101;
			bool b = physiological_average_gray[y * 256 + x][2] == 101;

			equal = equal && r && g && b;
		}
	}

	EXPECT_TRUE(equal) << "Tonemapping was not successful";
}

TEST(OpenGL, ImageProcessingTaskGaussian)
{
	using namespace Vcl::Graphics::ImageProcessing::OpenGL;
	using namespace Vcl::Graphics;

	// Instantiate the image processor
	ImageProcessor proc;

	// Prepare the input
	std::vector<std::array<unsigned char, 4>> input_image(81 * 81);
	std::vector<std::array<unsigned char, 4>> blurred_image_ref(81 * 81);
	std::vector<std::array<unsigned char, 4>> blurred_image(81 * 81);

	input_image.clear();
	blurred_image_ref.clear();
	for (int i = 0; i < sizeof(pattern); i += 3)
	{
		input_image.push_back({ pattern[i + 0], pattern[i + 1], pattern[i + 2], 0 });
		blurred_image_ref.push_back({ blurred_pattern[i + 0], blurred_pattern[i + 1], blurred_pattern[i + 2], 0 });
	}

	Runtime::TextureResource init_res;
	init_res.Width = 81;
	init_res.Height = 81;
	init_res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_res.Data = stdext::make_span(reinterpret_cast<const uint8_t*>(input_image.data()), input_image.size() * sizeof(decltype(input_image)::value_type));

	Runtime::Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R16G16B16A16_FLOAT;
	desc2d.ArraySize = 1;
	desc2d.Width = 81;
	desc2d.Height = 81;
	desc2d.MipLevels = 1;
	Runtime::OpenGL::Texture2D input{ desc2d, &init_res };

	Runtime::Texture2DDescription outDesc2d;
	outDesc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	outDesc2d.ArraySize = 1;
	outDesc2d.Width = 81;
	outDesc2d.Height = 81;
	outDesc2d.MipLevels = 1;
	auto output = std::make_shared<Runtime::OpenGL::Texture2D>(outDesc2d);

	// Task to test
	Gaussian task{ &proc };

	// Configure the input
	task.inputSlot(0)->setResource(&input);
	task.outputSlot(0)->setResource(output);

	// Memory barrier ensuring safe writes
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Execute the kernel
	proc.execute(&task);

	// Memory barrier ensuring safe reads
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Read the output
	auto& out_tex = static_cast<const Runtime::OpenGL::Texture2D&>(*output);
	out_tex.read(4*blurred_image.size(), blurred_image.data());

	bool equal = true;
	for (int y = 0; y < 81; y++)
	{
		for (int x = 0; x < 81; x++)
		{
			// Account for off-by-one OpenGL implementation based differences
			const auto cmp = [](unsigned char a, unsigned char b)
			{
				unsigned char res = 0;
				if (a >= b) res = a - b;
				else res = b - a;
				return res == 0 || res == 1;
			};
			bool r = cmp(blurred_image[y * 81 + x][0], blurred_image_ref[y * 81 + x][0]);
			bool g = cmp(blurred_image[y * 81 + x][1], blurred_image_ref[y * 81 + x][1]);
			bool b = cmp(blurred_image[y * 81 + x][2], blurred_image_ref[y * 81 + x][2]);

			equal = equal && r && g && b;
		}
	}

	EXPECT_TRUE(equal) << "Gaussian blur was not successful";
}

TEST(OpenGL, ImageProcessingSimpleGraph)
{
	using namespace Vcl::Graphics::ImageProcessing::OpenGL;
	using namespace Vcl::Graphics;

	// Instantiate the image processor
	ImageProcessor proc;

	// Prepare the input
	std::vector<std::array<unsigned char, 4>> input_image(81 * 81);
	std::vector<std::array<unsigned char, 4>> blurred_image_ref(81 * 81);
	std::vector<std::array<unsigned char, 4>> blurred_image(81 * 81);

	input_image.clear();
	blurred_image_ref.clear();
	for (int i = 0; i < sizeof(pattern); i += 3)
	{
		input_image.push_back({ pattern[i + 0], pattern[i + 1], pattern[i + 2], 0 });
		blurred_image_ref.push_back({ blurred_mapped_pattern[i + 0], blurred_mapped_pattern[i + 1], blurred_mapped_pattern[i + 2], 0 });
	}

	Runtime::TextureResource init_res;
	init_res.Width = 81;
	init_res.Height = 81;
	init_res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	init_res.Data = stdext::make_span(reinterpret_cast<const uint8_t*>(input_image.data()), input_image.size()*sizeof(decltype(input_image)::value_type));

	Runtime::Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R16G16B16A16_FLOAT;
	desc2d.ArraySize = 1;
	desc2d.Width = 81;
	desc2d.Height = 81;
	desc2d.MipLevels = 1;
	Runtime::OpenGL::Texture2D input{ desc2d, &init_res };

	// Task to test
	Gaussian blur{ &proc };
	Luminance lum{ &proc };
	Tonemap tonemap{ &proc };

	// Configure the input
	blur.inputSlot(0)->setResource(&input);
	lum.inputSlot(0)->setResource(&input);

	tonemap.inputSlot(0)->setSource(blur.outputSlot(0));
	tonemap.inputSlot(1)->setSource(lum.outputSlot(0));

	// Memory barrier ensuring safe writes
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Execute the kernel
	proc.execute(&tonemap);

	// Memory barrier ensuring safe reads
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

	// Read the output
	auto output = tonemap.outputSlot(0)->resource();
	auto& out_tex = static_cast<const Runtime::OpenGL::Texture2D&>(*output);
	out_tex.read(4 * blurred_image.size(), blurred_image.data());

	bool equal = true;
	for (int y = 0; y < 81; y++)
	{
		for (int x = 0; x < 81; x++)
		{
			// Account for off-by-one OpenGL implementation based differences
			const auto cmp = [](unsigned char a, unsigned char b)
			{
				unsigned char res = 0;
				if (a >= b) res = a - b;
				else res = b - a;
				return res == 0 || res == 1;
			};
			bool r = cmp(blurred_image[y * 81 + x][0], blurred_image_ref[y * 81 + x][0]);
			bool g = cmp(blurred_image[y * 81 + x][1], blurred_image_ref[y * 81 + x][1]);
			bool b = cmp(blurred_image[y * 81 + x][2], blurred_image_ref[y * 81 + x][2]);

			equal = equal && r && g && b;
		}
	}

	EXPECT_TRUE(equal) << "Image pipeline was not successful";
}
