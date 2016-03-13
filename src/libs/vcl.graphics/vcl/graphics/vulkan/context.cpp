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
#include <iostream>
#include <vector>

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/vulkan/device.h>

namespace Vcl { namespace Graphics { namespace Vulkan
{
	Context::Context(Device* dev, gsl::span<const char*> layers, gsl::span<const char*> extensions)
	: _physicalDevice(dev)
	{
		// Properties of the device
		VkDeviceCreateInfo dev_info;
		dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		dev_info.pNext = nullptr;
		dev_info.flags = 0;

		// Enable additional layers
		dev_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
		dev_info.ppEnabledLayerNames = layers.data();

		// Enable additional extensions
		dev_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		dev_info.ppEnabledExtensionNames = extensions.data();

		// Enable features
		dev_info.pEnabledFeatures = nullptr;
		
		// Queue info
		float queuePriorities[] = { 0.0f, 0.0f };

		VkDeviceQueueCreateInfo queue_info[1];
		queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info[0].pNext = nullptr;
		queue_info[0].flags = 0;
		queue_info[0].queueCount = 2;
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

		// Allocate different command pools.
		// For each queue family, we allocate three command pools:
		// 1) For in-frequently reset buffers
		// 2) For static command buffers that are never deleted or reset
		// 3) For short-lived command buffers
		_cmdPools.resize(1);

		VkCommandPoolCreateInfo cmd_pool_info;
		cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmd_pool_info.pNext = nullptr;
		cmd_pool_info.queueFamilyIndex = 0;

		cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		res = vkCreateCommandPool(_device, &cmd_pool_info, nullptr, &_cmdPools[0][int(CommandBufferType::Default)]);
		VclCheck(res == VK_SUCCESS, "Command pool was created.");

		cmd_pool_info.flags = 0;
		res = vkCreateCommandPool(_device, &cmd_pool_info, nullptr, &_cmdPools[0][int(CommandBufferType::Static)]);
		VclCheck(res == VK_SUCCESS, "Command pool was created.");

		cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		res = vkCreateCommandPool(_device, &cmd_pool_info, nullptr, &_cmdPools[0][int(CommandBufferType::Transient)]);
		VclCheck(res == VK_SUCCESS, "Command pool was created.");
	}

	Context::~Context()
	{
		// Cleanup the allocated data
		for (const auto& pools : _cmdPools)
		{
			vkDestroyCommandPool(_device, pools[0], nullptr);
			vkDestroyCommandPool(_device, pools[1], nullptr);
			vkDestroyCommandPool(_device, pools[2], nullptr);
		}

		vkDestroyPipelineCache(_device, _pipelineCache, nullptr);
		vkDestroyDevice(_device, nullptr);
	}

	VkQueue Context::queue(uint32_t idx)
	{
		VkQueue q;
		vkGetDeviceQueue(_device, 0, idx, &q);

		return q;
	}

	VkCommandPool Context::commandPool(uint32_t queueIdx, CommandBufferType type)
	{
		return _cmdPools[queueIdx][int(CommandBufferType::Transient)];
	}
}}}
