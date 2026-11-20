/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2016 Basil Fierz
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

// C++ standard library
#include <cstdint>

#ifdef VCL_VULKAN_SUPPORT

// Vulkan
#	include <vulkan/vulkan.h>

// VCL
#	include <vcl/core/span.h>
#	include <vcl/graphics/runtime/resource/shader.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {
	class Shader : public Runtime::Shader
	{
	public:
		Shader(VkDevice device, ShaderType type, int tag, stdext::span<const uint32_t> data);
		Shader(Shader&& rhs);
		virtual ~Shader();

		//! Convert the shader type to the Vulkan stage flag
		static VkShaderStageFlagBits convert(ShaderType type);

		//! \returns the size in bytes of the push constants the shader declares
		uint32_t pushConstantSize() const { return _pushConstantSize; }

		VkPipelineShaderStageCreateInfo getCreateInfo() const;

	private:
		//! Parse the SPIR-V data and collect the interface of the shader
		void reflect(stdext::span<const uint32_t> data);

		//! Device for which the shader was created
		VkDevice _device{ VK_NULL_HANDLE };

		//! Vulkan shader module
		VkShaderModule _shaderModule{ VK_NULL_HANDLE };

		//! Size of the push-constants
		uint32_t _pushConstantSize{ 0 };
	};
}}}}
#endif // VCL_VULKAN_SUPPORT
