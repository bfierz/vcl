/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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

// Include the relevant parts from the library
#include <vcl/graphics/runtime/vulkan/resource/shader.h>
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/platform.h>

// Test
#include "test.h"

// Additional shaders
#include "quad.vert.spv.h"
#include "quad.frag.spv.h"
const stdext::span<const uint32_t> QuadSpirvVS32{ reinterpret_cast<uint32_t*>(QuadSpirvVSData), QuadSpirvVSSize / 4 };
const stdext::span<const uint32_t> QuadSpirvFS32{ reinterpret_cast<uint32_t*>(QuadSpirvFSData), QuadSpirvFSSize / 4 };

class VulkanShaderTest : public VulkanTest
{
};

TEST_F(VulkanShaderTest, BuildSimpleSpirvGraphicsShaderProgram)
{
	using namespace Vcl::Graphics::Runtime;
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	// Compile the shader stages
	Runtime::Vulkan::Shader vs(*_context, ShaderType::VertexShader, 0, QuadSpirvVS32);
	Runtime::Vulkan::Shader fs(*_context, ShaderType::FragmentShader, 0, QuadSpirvFS32);

	EXPECT_TRUE(vs.getCreateInfo().module != 0);
	EXPECT_TRUE(fs.getCreateInfo().module != 0);
}
