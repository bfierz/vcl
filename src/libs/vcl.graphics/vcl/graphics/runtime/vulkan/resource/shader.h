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
#include <map>

#ifdef VCL_VULKAN_SUPPORT

// Vulkan
#include <vulkan/vulkan.h>

// VCL
#include <vcl/core/span.h>
#include <vcl/graphics/runtime/resource/shader.h>
#include <vcl/graphics/vulkan/descriptorset.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan
{
	class Shader : public Runtime::Shader
	{
		using DescriptorSetLayoutBinding = Graphics::Vulkan::DescriptorSetLayoutBinding;
		using PushConstantDescriptor = Graphics::Vulkan::PushConstantDescriptor;

	public:
		Shader(VkDevice device, ShaderType type, int tag, stdext::span<const uint32_t> data);
		Shader(Shader&& rhs);
		virtual ~Shader();

		//! Convert the shader type to the Vulkan stage flag
		static VkShaderStageFlagBits convert(ShaderType type);

		//! Access sets of the shader
		std::vector<uint32_t> sets() const;

		//! Access the bindings of a given set
		stdext::span<const DescriptorSetLayoutBinding> bindings(uint32_t set) const;

		//! Access the push-constants of the shader
		PushConstantDescriptor pushConstants() const {
			return _pushConstantRange;
		}

		VkPipelineShaderStageCreateInfo getCreateInfo() const;
		
	private:
		//! Parse the SPIR-V data and create the data binding table
		void createBindingTable(stdext::span<const uint32_t> data);

		//! Device for which the shader was created
		VkDevice _device;

		//! Vulkan shader module
		VkShaderModule _shaderModule;

		//! Size of the push-constants
		PushConstantDescriptor _pushConstantRange;

		//! Resources exposed in the shader
		std::map<uint32_t, std::vector<DescriptorSetLayoutBinding>> _bindings;
	};
}}}}
#endif // VCL_VULKAN_SUPPORT
