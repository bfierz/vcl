/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#include <vcl/graphics/vulkan/context.h>

// C++ standard library
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/vulkan/device.h>

namespace Vcl { namespace Graphics { namespace Vulkan {

	CommandPool::CommandPool(VkDevice device, uint32_t queue_index)
	: _device(device)
	{
		// Allocate different command pools.
		// For each queue family, we allocate three command pools:
		// 1) For in-frequently reset buffers
		// 2) For static command buffers that are never deleted or reset
		// 3) For short-lived command buffers

		VkResult res;
		VkCommandPoolCreateInfo cmd_pool_info;
		cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmd_pool_info.pNext = nullptr;
		cmd_pool_info.queueFamilyIndex = queue_index;

		cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		res = vkCreateCommandPool(device, &cmd_pool_info, nullptr, &_pools[int(CommandBufferType::Default)]);
		VclCheck(res == VK_SUCCESS, "Command pool was created.");

		cmd_pool_info.flags = 0;
		res = vkCreateCommandPool(device, &cmd_pool_info, nullptr, &_pools[int(CommandBufferType::Static)]);
		VclCheck(res == VK_SUCCESS, "Command pool was created.");

		cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		res = vkCreateCommandPool(device, &cmd_pool_info, nullptr, &_pools[int(CommandBufferType::Transient)]);
		VclCheck(res == VK_SUCCESS, "Command pool was created.");
	}
	CommandPool::CommandPool(CommandPool&& rhs) noexcept
	{
		*this = std::move(rhs);
	}
	CommandPool::~CommandPool()
	{
		for (int i = 0; i < 3; i++)
		{
			if (_pools[i] != nullptr)
			{
				vkDestroyCommandPool(_device, _pools[i], nullptr);
				_pools[i] = nullptr;
			}
		}
	}
	CommandPool& CommandPool::operator=(CommandPool&& rhs) noexcept
	{
		std::swap(_device, rhs._device);
		std::swap(_pools, rhs._pools);
		return *this;
	}
	VkCommandPool CommandPool::operator[](CommandBufferType type)
	{
		return _pools[int(type)];
	}
	void CommandPool::reset()
	{
		VkResult res;
		res = vkResetCommandPool(_device, _pools[int(CommandBufferType::Default)], 0);
		VclCheck(res == VK_SUCCESS, "Command pool was reset.");
		res = vkResetCommandPool(_device, _pools[int(CommandBufferType::Transient)], 0);
		VclCheck(res == VK_SUCCESS, "Command pool was reset.");
	}

	namespace {
		//! Feature the context can't work without
		template<typename Features>
		struct RequiredFeature
		{
			VkBool32 Features::*Member;
			const char* Name;
		};

#define VCL_VK_FEATURE(type, name) \
	{                              \
		&type::name, #name         \
	}

		// Arrays of resources indexed with values from push constants
		const RequiredFeature<VkPhysicalDeviceFeatures> RequiredCoreFeatures[] = {
			VCL_VK_FEATURE(VkPhysicalDeviceFeatures, shaderUniformBufferArrayDynamicIndexing),
			VCL_VK_FEATURE(VkPhysicalDeviceFeatures, shaderStorageBufferArrayDynamicIndexing),
			VCL_VK_FEATURE(VkPhysicalDeviceFeatures, shaderSampledImageArrayDynamicIndexing),
			VCL_VK_FEATURE(VkPhysicalDeviceFeatures, shaderStorageImageArrayDynamicIndexing),
		};

		// Descriptor indexing: a single, partially filled descriptor array, which is updated while in use
		const RequiredFeature<VkPhysicalDeviceVulkan12Features> RequiredVulkan12Features[] = {
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, descriptorIndexing),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, runtimeDescriptorArray),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, descriptorBindingPartiallyBound),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, descriptorBindingUpdateUnusedWhilePending),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, descriptorBindingUniformBufferUpdateAfterBind),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, descriptorBindingStorageBufferUpdateAfterBind),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, descriptorBindingSampledImageUpdateAfterBind),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, descriptorBindingStorageImageUpdateAfterBind),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, shaderUniformBufferArrayNonUniformIndexing),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, shaderStorageBufferArrayNonUniformIndexing),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, shaderSampledImageArrayNonUniformIndexing),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan12Features, shaderStorageImageArrayNonUniformIndexing),
		};

		const RequiredFeature<VkPhysicalDeviceVulkan13Features> RequiredVulkan13Features[] = {
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan13Features, synchronization2),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan13Features, dynamicRendering),
			VCL_VK_FEATURE(VkPhysicalDeviceVulkan13Features, maintenance4),
		};

#undef VCL_VK_FEATURE

		//! Enable a required feature, or record it as missing
		template<typename Features>
		void require(const Features& supported, Features& enabled, const RequiredFeature<Features>& feature, std::string& missing)
		{
			if (supported.*feature.Member)
				enabled.*feature.Member = VK_TRUE;
			else
				missing += std::string("\n  feature ") + feature.Name;
		}

		//! Feature structures of a device, chained for queries and device creation
		struct FeatureChain
		{
			FeatureChain()
			{
				Core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
				Vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
				Vk13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
				Mutable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT;
				Mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
			}

			//! Chain the structures. Structures of extensions which are not used are left out.
			//! \returns the head of the chain
			VkPhysicalDeviceFeatures2* link(bool with_mutable, bool with_mesh)
			{
				void** next = &Core.pNext;
				*next = &Vk12;
				next = &Vk12.pNext;
				*next = &Vk13;
				next = &Vk13.pNext;
				if (with_mutable)
				{
					*next = &Mutable;
					next = &Mutable.pNext;
				}
				if (with_mesh)
				{
					*next = &Mesh;
					next = &Mesh.pNext;
				}
				*next = nullptr;
				return &Core;
			}

			VkPhysicalDeviceFeatures2 Core = {};
			VkPhysicalDeviceVulkan12Features Vk12 = {};
			VkPhysicalDeviceVulkan13Features Vk13 = {};
			VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT Mutable = {};
			VkPhysicalDeviceMeshShaderFeaturesEXT Mesh = {};
		};

		//! \returns the features the device supports
		FeatureChain querySupportedFeatures(Device* dev)
		{
			FeatureChain supported;
			const bool with_mutable = dev->supportsExtension(VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME);
			const bool with_mesh = dev->supportsExtension(VK_EXT_MESH_SHADER_EXTENSION_NAME);
			vkGetPhysicalDeviceFeatures2(*dev, supported.link(with_mutable, with_mesh));
			return supported;
		}

		//! Select the features to enable.
		//! The bindless resource model requires descriptor indexing (core in Vulkan 1.2)
		//! and mutable descriptors. Fail early if the device doesn't provide them.
		//! \throws std::runtime_error listing the missing features
		FeatureChain enableFeatures(Device* dev, const FeatureChain& supported)
		{
			FeatureChain enabled;

			std::string missing;
			if (!dev->supportsExtension(VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME))
				missing += std::string("\n  extension ") + VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME;
			for (const auto& feature : RequiredCoreFeatures)
				require(supported.Core.features, enabled.Core.features, feature, missing);
			for (const auto& feature : RequiredVulkan12Features)
				require(supported.Vk12, enabled.Vk12, feature, missing);
			for (const auto& feature : RequiredVulkan13Features)
				require(supported.Vk13, enabled.Vk13, feature, missing);
			require(supported.Mutable, enabled.Mutable, RequiredFeature<VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT>{ &VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT::mutableDescriptorType, "mutableDescriptorType" }, missing);
			if (!missing.empty())
				throw std::runtime_error("Vulkan device '" + dev->name() + "' doesn't support the bindless resource model. Missing:" + missing);

			// Optional core features used by the pipeline states and samplers, if available
			const VkPhysicalDeviceFeatures& optional = supported.Core.features;
			VkPhysicalDeviceFeatures& features = enabled.Core.features;
			features.geometryShader = optional.geometryShader;
			features.tessellationShader = optional.tessellationShader;
			features.independentBlend = optional.independentBlend;
			features.dualSrcBlend = optional.dualSrcBlend;
			features.logicOp = optional.logicOp;
			features.fillModeNonSolid = optional.fillModeNonSolid;
			features.depthBiasClamp = optional.depthBiasClamp;
			features.samplerAnisotropy = optional.samplerAnisotropy;

			// Mesh shaders are optional. Task shaders are enabled together with them, if available.
			if (supported.Mesh.meshShader)
			{
				enabled.Mesh.meshShader = VK_TRUE;
				enabled.Mesh.taskShader = supported.Mesh.taskShader;
			}

			return enabled;
		}

		//! \returns the features of the device in terms of the shared descriptions
		Runtime::DeviceFeatures describeFeatures(Device* dev, const FeatureChain& enabled)
		{
			// Limits of descriptor sets, which are updated while in use
			VkPhysicalDeviceVulkan12Properties vk12_props = {};
			vk12_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
			VkPhysicalDeviceProperties2 props = {};
			props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			props.pNext = &vk12_props;
			vkGetPhysicalDeviceProperties2(*dev, &props);

			const VkPhysicalDeviceFeatures& features = enabled.Core.features;
			Runtime::DeviceFeatures desc;
			desc.BindlessResources = true;
			desc.GeometryShaders = features.geometryShader == VK_TRUE;
			// Tessellation needs the patch size, which the pipeline description doesn't provide
			desc.TessellationShaders = false;
			desc.MeshShaders = enabled.Mesh.meshShader == VK_TRUE;
			desc.TaskShaders = enabled.Mesh.taskShader == VK_TRUE;
			desc.IndependentBlend = features.independentBlend == VK_TRUE;
			desc.DualSourceBlend = features.dualSrcBlend == VK_TRUE;
			desc.LogicOp = features.logicOp == VK_TRUE;
			desc.WireframeFill = features.fillModeNonSolid == VK_TRUE;
			desc.DepthBiasClamp = features.depthBiasClamp == VK_TRUE;
			// Requires the 'depthClamp' feature, which is not enabled
			desc.DepthClipControl = false;
			desc.SamplerAnisotropy = features.samplerAnisotropy == VK_TRUE;
			desc.MaxRenderTargets = props.properties.limits.maxColorAttachments;
			desc.MaxPushConstantSize = props.properties.limits.maxPushConstantsSize;
			desc.MaxResourceHeapCapacity = std::min({ vk12_props.maxDescriptorSetUpdateAfterBindUniformBuffers,
													  vk12_props.maxDescriptorSetUpdateAfterBindStorageBuffers,
													  vk12_props.maxDescriptorSetUpdateAfterBindSampledImages,
													  vk12_props.maxDescriptorSetUpdateAfterBindStorageImages,
													  vk12_props.maxUpdateAfterBindDescriptorsInAllPools });
			desc.MaxSamplerHeapCapacity = std::min(vk12_props.maxDescriptorSetUpdateAfterBindSamplers, vk12_props.maxUpdateAfterBindDescriptorsInAllPools);
			return desc;
		}

		//! Add an extension to the list, unless it is already listed
		void addExtension(std::vector<const char*>& extensions, const char* name)
		{
			const auto is_name = [name](const char* ext) { return std::strcmp(ext, name) == 0; };
			if (std::none_of(extensions.begin(), extensions.end(), is_name))
				extensions.push_back(name);
		}
	}

	Context::Context(Device* dev, stdext::span<const char*> extensions)
	: _physicalDevice(dev)
	{
		// Collect information about the available command queues
		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(*dev, &queue_family_count, nullptr);
		_queueFamilies.resize(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(*dev, &queue_family_count, _queueFamilies.data());

		const FeatureChain supported = querySupportedFeatures(dev);
		FeatureChain enabled = enableFeatures(dev, supported);
		_enabledFeatures = enabled.Core.features;
		_meshShadersEnabled = enabled.Mesh.meshShader == VK_TRUE;
		_taskShadersEnabled = enabled.Mesh.taskShader == VK_TRUE;
		_features = describeFeatures(dev, enabled);

		// Enable additional extensions
		std::vector<const char*> enabled_extensions(extensions.begin(), extensions.end());
		addExtension(enabled_extensions, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME);
		if (_meshShadersEnabled)
			addExtension(enabled_extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);

		createDevice(enabled_extensions, enabled.link(true, _meshShadersEnabled));

		if (_meshShadersEnabled)
			_drawMeshTasks = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(_device, "vkCmdDrawMeshTasksEXT"));

		_cmdPools.emplace_back(_device, 0);
	}

	void Context::createDevice(const std::vector<const char*>& extensions, const VkPhysicalDeviceFeatures2* features)
	{
		// Properties of the device
		VkDeviceCreateInfo dev_info;
		dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		dev_info.flags = 0;

		// Zero deprecated feature
		dev_info.enabledLayerCount = 0;
		dev_info.ppEnabledLayerNames = nullptr;

		dev_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		dev_info.ppEnabledExtensionNames = extensions.data();

		dev_info.pEnabledFeatures = nullptr;
		dev_info.pNext = features;

		// Queue info
		float queuePriorities[] = { 0.0f, 0.0f };

		VkDeviceQueueCreateInfo queue_info[1];
		queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info[0].pNext = nullptr;
		queue_info[0].flags = 0;
		queue_info[0].queueCount = std::min(2u, _queueFamilies[0].queueCount);
		queue_info[0].queueFamilyIndex = 0;
		queue_info[0].pQueuePriorities = queuePriorities;

		dev_info.queueCreateInfoCount = 1;
		dev_info.pQueueCreateInfos = queue_info;

		VkResult res = vkCreateDevice(*_physicalDevice, &dev_info, nullptr, &_device);
		if (res != VkResult::VK_SUCCESS)
			throw "";

		VkPipelineCacheCreateInfo cache_info;
		cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		cache_info.pNext = nullptr;
		cache_info.flags = 0;
		cache_info.initialDataSize = 0;
		cache_info.pInitialData = nullptr;

		res = vkCreatePipelineCache(_device, &cache_info, nullptr, &_pipelineCache);
		if (res != VkResult::VK_SUCCESS)
			throw "";
	}

	Context::~Context()
	{
		_cmdPools.clear();
		vkDestroyPipelineCache(_device, _pipelineCache, nullptr);
		vkDestroyDevice(_device, nullptr);
	}

	VkQueue Context::queue(VkQueueFlags flags, uint32_t idx)
	{
		for (size_t q = 0; q < _queueFamilies.size(); q++)
		{
			const auto& fam = _queueFamilies[q];
			if (fam.queueCount > 0 && (fam.queueFlags & flags) != 0)
			{
				VkQueue queue;
				vkGetDeviceQueue(_device, q, idx, &queue);
				return queue;
			}
		}

		return nullptr;
	}

	VkCommandPool Context::commandPool(uint32_t queueIdx, CommandBufferType type)
	{
		return _cmdPools[queueIdx][type];
	}
}}}
