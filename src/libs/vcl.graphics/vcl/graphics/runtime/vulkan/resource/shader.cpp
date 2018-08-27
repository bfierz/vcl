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

// SPIR-V cross
#include <spirv_glsl.hpp>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan
{
	using DescriptorSetLayoutBinding = Graphics::Vulkan::DescriptorSetLayoutBinding;
	using DescriptorType = Graphics::Vulkan::DescriptorType;
	using ShaderStage = Graphics::Vulkan::ShaderStage;

	Shader::Shader(VkDevice device, ShaderType type, int tag, stdext::span<const uint32_t> data)
	: Runtime::Shader(type, tag)
	, _device(device)
	{
		createBindingTable(data);

		// Create the actual shader module
		VkShaderModuleCreateInfo info;
		info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		info.pNext = nullptr;

		info.codeSize = data.size()*sizeof(uint32_t);
		info.pCode = data.data();
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

	std::vector<uint32_t> Shader::sets() const
	{
		std::vector<uint32_t> keys;
		keys.reserve(_bindings.size());
		std::transform(std::begin(_bindings), std::end(_bindings), std::back_inserter(keys), [](const auto& key_bindings)
		{
			return key_bindings.first;
		});
		return keys;
	}

	stdext::span<const DescriptorSetLayoutBinding> Shader::bindings(uint32_t set) const
	{
		const auto bindings_of_set = _bindings.find(set);
		if (bindings_of_set != _bindings.end())
		{
			return bindings_of_set->second;
		}

		return {};
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

	void Shader::createBindingTable(stdext::span<const uint32_t> data)
	{
		// Create the reflection data
		const spirv_cross::CompilerGLSL spir_mod({ data.begin(), data.end() });
		const spirv_cross::ShaderResources resources = spir_mod.get_shader_resources();

		const auto curr_shader_type = [t=type()]()
		{
			switch (t)
			{
			case ShaderType::VertexShader:     return Flags<ShaderStage>(ShaderStage::Vertex);
			case ShaderType::ControlShader:    return Flags<ShaderStage>(ShaderStage::TessellationControl);
			case ShaderType::EvaluationShader: return Flags<ShaderStage>(ShaderStage::TessellationEvaluation);
			case ShaderType::GeometryShader:   return Flags<ShaderStage>(ShaderStage::Geometry);
			case ShaderType::FragmentShader:   return Flags<ShaderStage>(ShaderStage::Fragment);
			case ShaderType::ComputeShader:    return Flags<ShaderStage>(ShaderStage::Compute);
			default: { VclDebugError("Enumeration value is valid."); }
			}
			return Flags<ShaderStage>();
		}();

		// Process the push constants
		VclCheck(resources.push_constant_buffers.size() <= 1, 
		         "Only one push-constant buffer supported per shader stage");
		for (const auto& resource : resources.push_constant_buffers)
		{
			using Graphics::Vulkan::convert;

			const auto offset = spir_mod.get_decoration(resource.id, spv::DecorationOffset);
			const auto& type = spir_mod.get_type(resource.type_id);
			const auto size = spir_mod.get_declared_struct_size(type);
			_pushConstantRange.StageFlags = curr_shader_type;
			_pushConstantRange.Offset = 0;
			_pushConstantRange.Size = size;
		}

		// Process the shader resource bindings
		_bindings.clear();
		for (const auto& resource : resources.uniform_buffers)
		{
			const auto set = spir_mod.get_decoration(resource.id, spv::DecorationDescriptorSet);
			const auto binding = spir_mod.get_decoration(resource.id, spv::DecorationBinding);
			const auto& type = spir_mod.get_type(resource.type_id);
			uint32_t array_size = 1;
			if (type.array.size() > 0)
			{
				array_size = type.array[0];
			}
			_bindings[set].emplace_back(DescriptorSetLayoutBinding{ DescriptorType::UniformBuffer, array_size, curr_shader_type, binding });
		}
		for (const auto& resource : resources.storage_buffers)
		{
			const auto set = spir_mod.get_decoration(resource.id, spv::DecorationDescriptorSet);
			const auto binding = spir_mod.get_decoration(resource.id, spv::DecorationBinding);
			const auto& type = spir_mod.get_type(resource.type_id);
			uint32_t array_size = 1;
			if (type.array.size() > 0)
			{
				array_size = type.array[0];
			}
			_bindings[set].emplace_back(DescriptorSetLayoutBinding{ DescriptorType::StorageBuffer, array_size, curr_shader_type, binding });
		}

		for (auto& set : _bindings)
		{
			std::sort(std::begin(set.second), std::end(set.second), [](const auto& a, const auto& b)
			{
				return a.Binding < b.Binding;
			});
		}
	}
}}}}
#endif // VCL_VULKAN_SUPPORT
