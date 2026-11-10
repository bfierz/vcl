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
	}

	Context::Context(Device* dev, stdext::span<const char*> extensions)
	: _physicalDevice(dev)
	{
		// Collect information about the available command queues
		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(*dev, &queue_family_count, nullptr);
		_queueFamilies.resize(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(*dev, &queue_family_count, _queueFamilies.data());

		// The bindless resource model requires descriptor indexing (core in Vulkan 1.2)
		// and mutable descriptors. Fail early if the device doesn't provide them.
		VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT supported_mutable = {};
		supported_mutable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT;
		VkPhysicalDeviceVulkan13Features supported_vk13 = {};
		supported_vk13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
		supported_vk13.pNext = &supported_mutable;
		VkPhysicalDeviceVulkan12Features supported_vk12 = {};
		supported_vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		supported_vk12.pNext = &supported_vk13;
		VkPhysicalDeviceFeatures2 supported = {};
		supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		supported.pNext = &supported_vk12;

		const bool has_mutable_descriptors = dev->supportsExtension(VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME);
		if (!has_mutable_descriptors)
			supported_vk13.pNext = nullptr;
		vkGetPhysicalDeviceFeatures2(*dev, &supported);

		VkPhysicalDeviceFeatures2 features = {};
		features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		VkPhysicalDeviceVulkan12Features vk12_features = {};
		vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		VkPhysicalDeviceVulkan13Features vk13_features = {};
		vk13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
		VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mutable_features = {};
		mutable_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT;
		features.pNext = &vk12_features;
		vk12_features.pNext = &vk13_features;
		vk13_features.pNext = &mutable_features;

		std::string missing;
		if (!has_mutable_descriptors)
			missing += std::string("\n  extension ") + VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME;
		for (const auto& feature : RequiredCoreFeatures)
			require(supported.features, features.features, feature, missing);
		for (const auto& feature : RequiredVulkan12Features)
			require(supported_vk12, vk12_features, feature, missing);
		for (const auto& feature : RequiredVulkan13Features)
			require(supported_vk13, vk13_features, feature, missing);
		require(supported_mutable, mutable_features, RequiredFeature<VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT>{ &VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT::mutableDescriptorType, "mutableDescriptorType" }, missing);
		if (!missing.empty())
			throw std::runtime_error("Vulkan device '" + dev->name() + "' doesn't support the bindless resource model. Missing:" + missing);

		// Enable the optional core features used by the pipeline states and samplers, if available
		const VkPhysicalDeviceFeatures& supported_features = supported.features;
		features.features.geometryShader = supported_features.geometryShader;
		features.features.tessellationShader = supported_features.tessellationShader;
		features.features.independentBlend = supported_features.independentBlend;
		features.features.dualSrcBlend = supported_features.dualSrcBlend;
		features.features.logicOp = supported_features.logicOp;
		features.features.fillModeNonSolid = supported_features.fillModeNonSolid;
		features.features.depthBiasClamp = supported_features.depthBiasClamp;
		features.features.samplerAnisotropy = supported_features.samplerAnisotropy;
		_enabledFeatures = features.features;

		// Enable additional extensions
		std::vector<const char*> enabled_extensions(extensions.begin(), extensions.end());
		const auto is_mutable_ext = [](const char* name) { return std::strcmp(name, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME) == 0; };
		if (std::none_of(enabled_extensions.begin(), enabled_extensions.end(), is_mutable_ext))
			enabled_extensions.push_back(VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME);

		// Properties of the device
		VkDeviceCreateInfo dev_info;
		dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		dev_info.flags = 0;

		// Zero deprecated feature
		dev_info.enabledLayerCount = 0;
		dev_info.ppEnabledLayerNames = nullptr;

		dev_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
		dev_info.ppEnabledExtensionNames = enabled_extensions.data();

		dev_info.pEnabledFeatures = nullptr;
		dev_info.pNext = &features;

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

		_cmdPools.emplace_back(_device, 0);
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
