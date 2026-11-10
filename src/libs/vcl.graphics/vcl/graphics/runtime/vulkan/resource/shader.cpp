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
#	include <spirv_glsl.hpp>

// VCL
#	include <vcl/core/contract.h>
#	include <vcl/core/enummap.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace Vulkan {
	namespace {
		// clang-format off
		constexpr EnumMapping<ShaderType, VkShaderStageFlagBits> StageBitsTable[] = {
			{ ShaderType::VertexShader,     VK_SHADER_STAGE_VERTEX_BIT },
			{ ShaderType::ControlShader,    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT },
			{ ShaderType::EvaluationShader, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT },
			{ ShaderType::GeometryShader,   VK_SHADER_STAGE_GEOMETRY_BIT },
			{ ShaderType::FragmentShader,   VK_SHADER_STAGE_FRAGMENT_BIT },
			{ ShaderType::ComputeShader,    VK_SHADER_STAGE_COMPUTE_BIT },
			{ ShaderType::TaskShader,       VK_SHADER_STAGE_TASK_BIT_EXT },
			{ ShaderType::MeshShader,       VK_SHADER_STAGE_MESH_BIT_EXT },
		};
		// clang-format on
		constexpr EnumMap<ShaderType, VkShaderStageFlagBits, enumMapSize(StageBitsTable)> StageBits{ StageBitsTable };
	}

	Shader::Shader(VkDevice device, ShaderType type, int tag, stdext::span<const uint32_t> data)
	: Runtime::Shader(type, tag)
	, _device(device)
	{
		reflect(data);

		// Create the actual shader module
		VkShaderModuleCreateInfo info;
		info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		info.pNext = nullptr;

		info.codeSize = data.size() * sizeof(uint32_t);
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
		std::swap(_pushConstantSize, rhs._pushConstantSize);
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
		return StageBits[type];
	}

	void Shader::reflect(stdext::span<const uint32_t> data)
	{
		// Resources are accessed through the bindless descriptor heap, which has
		// a fixed layout. Only the push constants depend on the shader.
		const spirv_cross::CompilerGLSL spir_mod({ data.begin(), data.end() });
		const spirv_cross::ShaderResources resources = spir_mod.get_shader_resources();

		VclCheck(resources.push_constant_buffers.size() <= 1, "Only one push-constant buffer supported per shader stage");
		for (const auto& resource : resources.push_constant_buffers)
		{
			const auto& type = spir_mod.get_type(resource.base_type_id);
			_pushConstantSize = static_cast<uint32_t>(spir_mod.get_declared_struct_size(type));
		}
	}
}}}}
#endif // VCL_VULKAN_SUPPORT
