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
#include <vcl/graphics/runtime/vulkan/resource/shader.h>

// C++ standard library
#include <vector>

#ifdef VCL_VULKAN_SUPPORT

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan
{
	Shader::Shader(VkDevice device, ShaderType type, int tag, const char* source, size_t size)
	: Runtime::Shader(type, tag)
	, _device(device)
	{
		VkShaderModuleCreateInfo info;
		info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		info.pNext = nullptr;

		info.codeSize = size;
		info.pCode = reinterpret_cast<const uint32_t*>(source);
		info.flags = 0;


		VkResult res = vkCreateShaderModule(_device, &info, nullptr, &_shaderModule);
		VclEnsure(res == VK_SUCCESS, "Shader module was created.");
	}

	Shader::Shader(Shader&& rhs)
	: Runtime::Shader(rhs)
	{
		std::swap(_device, rhs._device);
		std::swap(_shaderModule, rhs._shaderModule);
	}

	Shader::~Shader()
	{
		if (_shaderModule)
			vkDestroyShaderModule(_device, _shaderModule, nullptr);
	}

	VkPipelineShaderStageCreateInfo Shader::getCreateInfo() const
	{
		VkPipelineShaderStageCreateInfo stage = {};
		stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage = convert(type());
		stage.module = _shaderModule;
		stage.pName = "main";
		return stage;
	}

	VkShaderStageFlagBits Shader::convert(ShaderType type)
	{
		switch (type)
		{
		case ShaderType::VertexShader:     return VK_SHADER_STAGE_VERTEX_BIT;
		case ShaderType::ControlShader:    return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
		case ShaderType::EvaluationShader: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
		case ShaderType::GeometryShader:   return VK_SHADER_STAGE_GEOMETRY_BIT;
		case ShaderType::FragmentShader:   return VK_SHADER_STAGE_FRAGMENT_BIT;
		case ShaderType::ComputeShader:    return VK_SHADER_STAGE_COMPUTE_BIT;
		default: { VclDebugError("Enumeration value is valid."); }
		}

		return VK_SHADER_STAGE_ALL;
	}
}}}}
#endif // VCL_VULKAN_SUPPORT
