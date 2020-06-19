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

// VCL configuration
#include <vcl/config/global.h>

// Include the relevant parts from the library
#include <vcl/graphics/d3d12/d3d.h>
#include <vcl/graphics/d3d12/semaphore.h>
#include <vcl/graphics/runtime/d3d12/resource/texture.h>
#include <vcl/math/ceil.h>

// Google test
#include <gtest/gtest.h>

extern std::unique_ptr<Vcl::Graphics::D3D12::Device> device;

std::vector<unsigned char> createTestPattern(int width, int height, int depth)
{
	std::array<unsigned char, 12> pattern =
	{
		0xff, 0x00, 0x00, 0xff,
		0x00, 0xff, 0x00, 0xff,
		0x00, 0x00, 0xff, 0xff,
	};

	std::vector<unsigned char> image;
	image.reserve(width*height*depth*4);
	for (int d = 0; d < depth; d++)
	for (int h = 0; h < height; h++)
	for (int w = 0; w < width*4; w++)
	{
		image.push_back(pattern[4*(h%3) + w%4]);
	}
	return image;
}

void verifySize(const Vcl::Graphics::Runtime::D3D12::Texture& tex, int exp_w, int exp_h, int exp_d)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	const auto desc = tex.handle()->GetDesc();

	size_t w = desc.Width;
	size_t h = desc.Height;
	size_t d = desc.DepthOrArraySize;

	EXPECT_EQ(exp_w, w) << "Texture has wrong width. Expected " << exp_w << ", got " << w;
	EXPECT_EQ(exp_h, h) << "Texture has wrong height count. Expected " << exp_h << ", got " << h;
	EXPECT_EQ(exp_d, d) << "Texture has wrong depth/layer count. Expected " << exp_d << ", got " << d;
}

void verifyContent(ID3D12CommandQueue* cmd_queue, ID3D12GraphicsCommandList* cmd_list, Vcl::Graphics::Runtime::D3D12::Texture& tex, stdext::span<const unsigned char> test_image)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;
	using Vcl::Mathematics::ceil;

	const uint32_t w = tex.width();
	const uint32_t h = tex.height();
	const uint32_t d = tex.depth();
	const uint32_t l = tex.layers();
	const uint32_t pixel_size = Vcl::Graphics::sizeInBytes(tex.format());
	const uint32_t row_pitch = ceil<256>(w * pixel_size);
	const uint32_t slice_pitch = ceil<512>(h * row_pitch);

	BufferDescription read_back_desc =
	{
		std::max(l * slice_pitch, d * h * row_pitch),
		BufferUsage::CopyDst | BufferUsage::MapRead
	};
	Runtime::D3D12::Buffer buf(device.get(), read_back_desc);

	tex.copyTo(cmd_list, buf);
	VCL_DIRECT3D_SAFE_CALL(cmd_list->Close());

	ID3D12CommandList* const generic_cmd_list = cmd_list;
	cmd_queue->ExecuteCommandLists(1, &generic_cmd_list);

	Vcl::Graphics::D3D12::Semaphore sema{ device->nativeDevice() };
	const auto sig_value = sema.signal(cmd_queue);
	sema.wait(sig_value);

	bool equal = true;
	auto ptr = (unsigned char*)buf.map({ 0, read_back_desc.SizeInBytes });
	for (unsigned int i = 0; i < l; i++)
	{
		auto img_ptr = ptr + i * slice_pitch;
		for (unsigned int z = 0; z < d; z++)
		for (unsigned int y = 0; y < h; y++, img_ptr += row_pitch)
		for (unsigned int x = 0; x < pixel_size * w; x++)
		{
			equal = equal && (img_ptr[x] == test_image[pixel_size * w * y + x]);
		}
	}
	buf.unmap({ 0, 0 });
	EXPECT_TRUE(equal) << "Initialisation data is correct.";
}

TEST(D3D12Texture, InitEmptyTexture1D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture1DDescription desc1d;
	desc1d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc1d.ArraySize = 1;
	desc1d.Width = 32;
	desc1d.MipLevels = 1;
	Runtime::D3D12::Texture1D tex{ device.get(), desc1d };
	
	verifySize(tex, 32, 1, 1);
}

TEST(D3D12Texture, InitTexture1D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture1DDescription desc1d;
	desc1d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc1d.Usage = TextureUsage::CopySrc;
	desc1d.ArraySize = 1;
	desc1d.Width = 32;
	desc1d.MipLevels = 1;

	const auto test_image = createTestPattern(32, 1, 1);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Width = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;

	auto cmd_queue = device->createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	Runtime::D3D12::Texture1D tex{ device.get(), desc1d, &res, cmd_list.Get() };
	verifyContent(cmd_queue.Get(), cmd_list.Get(), tex, stdext::make_span(test_image));
}

/*TEST(D3D12Texture, ClearTexture1D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture1DDescription desc1d;
	desc1d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc1d.ArraySize = 1;
	desc1d.Width = 32;
	desc1d.MipLevels = 1;

	const auto test_image = createTestPattern(32, 1, 1);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Width = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;
	Runtime::D3D12::Texture1D tex{ desc1d, &res };

	std::vector<unsigned char> image(32 * 1 * 4);
	tex.read(image.size(), image.data());
	EXPECT_EQ(image, test_image);

	int zero = 0;
	tex.clear(SurfaceFormat::R8G8B8A8_UNORM, &zero);

	tex.read(image.size(), image.data());
	EXPECT_TRUE(std::all_of(image.begin(), image.end(), [](unsigned char c) { return c == 0; }));
}*/

TEST(D3D12Texture, InitEmptyTexture1DArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture1DDescription desc1d;
	desc1d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc1d.ArraySize = 2;
	desc1d.Width = 32;
	desc1d.MipLevels = 1;
	Runtime::D3D12::Texture1DArray tex{ device.get(), desc1d };

	verifySize(tex, 32, 1, 2);
}

TEST(D3D12Texture, InitTexture1DArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture1DDescription desc1d;
	desc1d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc1d.Usage = TextureUsage::CopySrc;
	desc1d.ArraySize = 2;
	desc1d.Width = 32;
	desc1d.MipLevels = 1;

	const auto test_image = createTestPattern(32, 1, 2);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Layers = 2;
	res.Width = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;

	auto cmd_queue = device->createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	Runtime::D3D12::Texture1DArray tex{ device.get(), desc1d, &res, cmd_list.Get() };
	verifyContent(cmd_queue.Get(), cmd_list.Get(), tex, stdext::make_span(test_image));
}

TEST(D3D12Texture, InitEmptyTexture2D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 1;
	desc2d.Width = 32;
	desc2d.Height = 32;
	desc2d.MipLevels = 1;
	Runtime::D3D12::Texture2D tex{ device.get(), desc2d };

	verifySize(tex, 32, 32, 1);
}

TEST(D3D12Texture, InitTexture2D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	const unsigned int size = 32;

	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.Usage = TextureUsage::CopySrc;
	desc2d.ArraySize = 1;
	desc2d.Width = size;
	desc2d.Height = size;
	desc2d.MipLevels = 1;

	const auto test_image = createTestPattern(size, size, 1);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Width = size;
	res.Height = size;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;

	auto cmd_queue = device->createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	Runtime::D3D12::Texture2D tex{ device.get(), desc2d, &res, cmd_list.Get() };
	verifyContent(cmd_queue.Get(), cmd_list.Get(), tex, stdext::make_span(test_image));
}

TEST(D3D12Texture, InitEmptyTexture2DArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.ArraySize = 2;
	desc2d.Width = 32;
	desc2d.Height = 32;
	desc2d.MipLevels = 1;
	Runtime::D3D12::Texture2DArray tex{ device.get(), desc2d };

	verifySize(tex, 32, 32, 2);
}

TEST(D3D12Texture, InitTexture2DArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture2DDescription desc2d;
	desc2d.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc2d.Usage = TextureUsage::CopySrc;
	desc2d.ArraySize = 2;
	desc2d.Width = 32;
	desc2d.Height = 32;
	desc2d.MipLevels = 1;

	const auto test_image = createTestPattern(32, 32, 2);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Layers = 2;
	res.Width = 32;
	res.Height = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;

	auto cmd_queue = device->createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	Runtime::D3D12::Texture2DArray tex{ device.get(), desc2d, &res, cmd_list.Get() };
	verifyContent(cmd_queue.Get(), cmd_list.Get(), tex, stdext::make_span(test_image));
}

TEST(D3D12Texture, InitEmptyTexture3D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture3DDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.Width = 32;
	desc.Height = 32;
	desc.Depth = 2;
	desc.MipLevels = 1;
	Runtime::D3D12::Texture3D tex{ device.get(), desc };

	verifySize(tex, 32, 32, 2);
}

TEST(D3D12Texture, InitTexture3D)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	Texture3DDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.Usage = TextureUsage::CopySrc;
	desc.Width = 32;
	desc.Height = 32;
	desc.Depth = 2;
	desc.MipLevels = 1;

	const auto test_image = createTestPattern(32, 32, 2);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Width = 32;
	res.Height = 32;
	res.Depth = 2;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;

	auto cmd_queue = device->createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	Runtime::D3D12::Texture3D tex{ device.get(), desc, &res, cmd_list.Get() };
	verifyContent(cmd_queue.Get(), cmd_list.Get(), tex, stdext::make_span(test_image));
}

TEST(D3D12Texture, InitEmptyTextureCube)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	TextureCubeDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.ArraySize = 1;
	desc.Width = 32;
	desc.Height = 32;
	desc.MipLevels = 1;
	Runtime::D3D12::TextureCube tex{ device.get(), desc };

	verifySize(tex, 32, 32, 6);
}

TEST(D3D12Texture, InitTextureCube)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	TextureCubeDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.Usage = TextureUsage::CopySrc;
	desc.ArraySize = 1;
	desc.Width = 32;
	desc.Height = 32;
	desc.MipLevels = 1;

	const auto test_image = createTestPattern(32, 32, 6);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Width = 32;
	res.Height = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;

	auto cmd_queue = device->createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	Runtime::D3D12::TextureCube tex{ device.get(), desc, &res, cmd_list.Get() };
	verifyContent(cmd_queue.Get(), cmd_list.Get(), tex, stdext::make_span(test_image));
}

TEST(D3D12Texture, InitEmptyTextureCubeArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	TextureCubeDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.ArraySize = 2;
	desc.Width = 32;
	desc.Height = 32;
	desc.MipLevels = 1;
	Runtime::D3D12::TextureCubeArray tex{ device.get(), desc };

	verifySize(tex, 32, 32, 12);
}

TEST(D3D12Texture, InitTextureCubeArray)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics;

	TextureCubeDescription desc;
	desc.Format = SurfaceFormat::R8G8B8A8_UNORM;
	desc.Usage = TextureUsage::CopySrc;
	desc.ArraySize = 2;
	desc.Width = 32;
	desc.Height = 32;
	desc.MipLevels = 1;

	const auto test_image = createTestPattern(32, 32, 12);
	TextureResource res;
	res.Data = stdext::make_span(test_image);
	res.Layers = 2;
	res.Width = 32;
	res.Height = 32;
	res.Format = SurfaceFormat::R8G8B8A8_UNORM;

	auto cmd_queue = device->createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_allocator = device->createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
	auto cmd_list = device->createCommandList(cmd_allocator.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
	Runtime::D3D12::TextureCubeArray tex{ device.get(), desc, &res, cmd_list.Get() };
	verifyContent(cmd_queue.Get(), cmd_list.Get(), tex, stdext::make_span(test_image));
}
