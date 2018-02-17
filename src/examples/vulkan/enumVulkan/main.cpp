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

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <iostream>
#include <vector>

// Vulkan API
#include <vulkan/vulkan.h>

// GLFW
#include <GLFW/glfw3.h>

// VCL
#include <vcl/graphics/runtime/vulkan/resource/buffer.h>
#include <vcl/graphics/runtime/vulkan/resource/shader.h>
#include <vcl/graphics/vulkan/commands.h>
#include <vcl/graphics/vulkan/pipelinestate.h>
#include <vcl/graphics/vulkan/platform.h>
#include <vcl/graphics/vulkan/swapchain.h>
#include <vcl/graphics/vulkan/tools.h>

// Force the use of the NVIDIA GPU in an Optimius system
#ifdef VCL_COMPILER_MSVC
extern "C"
{
	_declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
}
#endif

// Load a binary file into a buffer (e.g. SPIR-V)
char *readBinaryFile(const char *filename, size_t *psize)
{
	long int size;
	size_t retval;
	void *shader_code;

	FILE *fp = fopen(filename, "rb");
	if (!fp) return NULL;

	fseek(fp, 0L, SEEK_END);
	size = ftell(fp);

	fseek(fp, 0L, SEEK_SET);

	shader_code = malloc(size);
	retval = fread(shader_code, size, 1, fp);
	assert(retval == 1);

	*psize = size;

	return (char*)shader_code;
}

void buildCommandBuffers(Vcl::Graphics::Vulkan::Backbuffer* bb, VkCommandPool cmdPool, VkRenderPass renderPass, uint32_t width, uint32_t height, std::vector<Vcl::Graphics::Vulkan::CommandBuffer>& drawCmdBuffers)
{
	drawCmdBuffers.reserve(bb->swapChain()->nrImages());
	for (int i = 0; i < bb->swapChain()->nrImages(); i++)
	{
		drawCmdBuffers.emplace_back(*bb->context(), cmdPool);
	}

	VkClearValue clearValues[2];
	clearValues[0].color = { 1.0f, 0.0f, 1.0f, 1.0f };
	clearValues[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.pNext = NULL;
	renderPassBeginInfo.renderPass = renderPass;
	renderPassBeginInfo.renderArea.offset.x = width / 4;
	renderPassBeginInfo.renderArea.offset.y = height / 4;
	renderPassBeginInfo.renderArea.extent.width = width / 2;
	renderPassBeginInfo.renderArea.extent.height = height / 2;
	renderPassBeginInfo.clearValueCount = 2;
	renderPassBeginInfo.pClearValues = clearValues;

	VkResult err;

	for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
	{
		// Set target frame buffer
		renderPassBeginInfo.framebuffer = bb->framebuffer(i);

		drawCmdBuffers[i].begin();
		drawCmdBuffers[i].beginRenderPass(&renderPassBeginInfo);

		// Update dynamic viewport state
		VkViewport viewport = {};
		viewport.x = (float)width / 4;
		viewport.y = (float)height / 4;
		viewport.height = (float)height / 2;
		viewport.width = (float)width / 2;
		viewport.minDepth = (float) 0.0f;
		viewport.maxDepth = (float) 1.0f;
		drawCmdBuffers[i].setViewport(0, { &viewport, 1 });

		// Update dynamic scissor state
		VkRect2D scissor = {};
		scissor.extent.width = width / 2;
		scissor.extent.height = height / 2;
		scissor.offset.x = width / 4;
		scissor.offset.y = height / 4;
		drawCmdBuffers[i].setScissor(0, { &scissor, 1 });

		//// Bind descriptor sets describing shader binding points
		//vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
		//
		//// Bind the rendering pipeline (including the shaders)
		//vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.solid);
		//
		//// Bind triangle vertices
		//VkDeviceSize offsets[1] = { 0 };
		//vkCmdBindVertexBuffers(drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, &vertices.buf, offsets);
		//
		//// Bind triangle indices
		//vkCmdBindIndexBuffer(drawCmdBuffers[i], indices.buf, 0, VK_INDEX_TYPE_UINT32);
		//
		//// Draw indexed triangle
		//vkCmdDrawIndexed(drawCmdBuffers[i], indices.count, 1, 0, 0, 1);

		drawCmdBuffers[i].endRenderPass();

		// Preparing the frame buffer for presentation
		drawCmdBuffers[i].prepareForPresent(bb->swapChain()->image(i));

		// Finalizing the command buffer
		drawCmdBuffers[i].end();
	}
}

class GlfwInstance
{
public:
	GlfwInstance()
	{
		glfwSetErrorCallback(errorCallback);
		_isInitialized = glfwInit() != 0 ? true : false;
		if (!_isInitialized)
			throw std::runtime_error{ "Unexpected error when initializing GLFW" };

		// Since we are using vulkan we do not need any predefined client API
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		// Check Vulkan support
		_vulkanSupport = glfwVulkanSupported() != 0 ? true : false;
		
		unsigned int nr_exts = 0;
		auto exts = glfwGetRequiredInstanceExtensions(&nr_exts);
		_vulkanExtensions = { exts, nr_exts };
	}

	~GlfwInstance()
	{
		if (_isInitialized)
			glfwTerminate();
	}

public:
	//! \returns true if vulkan is supported
	bool isVulkanSupported() const { return _vulkanSupport; }

	//! \returns the required vulkan extensions
	gsl::span<const char*> vulkanExtensions() const { return _vulkanExtensions; }

public:
	void createSurface();

private:
	static void errorCallback(int error, const char* description)
	{
		fprintf(stderr, "Error: %s\n", description);
	}

private:
	bool _isInitialized{ false };

	//! Query if Vulkan is supported
	bool _vulkanSupport{ false };

	//! Required Vulkan extensions
	gsl::span<const char*> _vulkanExtensions;
};


int main(int argc, char* argv[])
{
	using namespace Vcl::Graphics::Vulkan;

	GlfwInstance glfw;
	if (!glfw.isVulkanSupported())
	{
		std::cerr << "Vulkan is not supported" << std::endl;
		return 1;
	}

	// Vulkan extension
	std::array<const char*, 1> context_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	// Initialize the Vulkan platform
	auto platform = std::make_unique<Platform>(glfw.vulkanExtensions());
	auto& device = platform->device(0);
	auto context = device.createContext(context_extensions);
	CommandQueue queue{ context.get(), 0 };

	// Create a window
	auto window = glfwCreateWindow(1280, 720, "Vulkan Demo", nullptr, nullptr);
	if (!window)
	{
		std::cerr << "Cannot create a window in which to draw!" << std::endl;
		return 1;
	}
	
	// Begin: Setup surface and swap-chain

	// Create a WSI surface for the window
	VkSurfaceKHR surface_ctx;
	glfwCreateWindowSurface(*platform, window, nullptr, &surface_ctx);

	Vcl::Graphics::Vulkan::BasicSurfaceDescription desc;
	desc.Surface = surface_ctx;
	desc.NumberOfImages = 4;
	desc.ColourFormat = VK_FORMAT_B8G8R8A8_UNORM;
	desc.DepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
	desc.Width = 1280;
	desc.Height = 720;
	auto surface = createBasicSurface(*platform, *context, queue, desc);

	VkRenderPass render_pass = createDefaultRenderPass(*context, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT);

	// End: Setup surface and swap-chain

	// Begin: Scene setup
	/*
	// Setup buffers
	Vcl::Graphics::Runtime::BufferDescription staging_buffer_desc;
	staging_buffer_desc.CPUAccess = Vcl::Graphics::Runtime::ResourceAccess::Write | Vcl::Graphics::Runtime::ResourceAccess::Read;
	staging_buffer_desc.SizeInBytes = 4096;
	staging_buffer_desc.Usage = Vcl::Graphics::Runtime::ResourceUsage::Staging;
	Vcl::Graphics::Runtime::Vulkan::Buffer staging_buffer
	{
		context.get(), staging_buffer_desc, Vcl::Graphics::Runtime::Vulkan::BufferUsage::TransferSource
	};
	void* mapped_ptr = staging_buffer.memory()->map(0, 4096);
	staging_buffer.memory()->unmap();

	// Descriptor set
	DescriptorSetLayout descriptor_set_layout = 
	{
		context.get(),
		{
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0 }
		}
	};

	// Create a descriptor-set and pipeline layout
	PipelineLayout layout{ context.get(), &descriptor_set_layout };

	// Create the pipeline state	
	Vcl::Graphics::Runtime::InputLayoutDescription opaqueTetraLayout =
	{
		{
			{ 0, sizeof(Eigen::Vector4i), Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerObject },
			{ 1, sizeof(Eigen::Vector4f), Vcl::Graphics::Runtime::VertexDataClassification::VertexDataPerObject }
		},
		{
			{ "Position",  Vcl::Graphics::SurfaceFormat::R32G32B32_FLOAT, 0, 0, 0 },
			{ "Colour", Vcl::Graphics::SurfaceFormat::R32G32B32_FLOAT, 0, 1, 0 }
		}
	};
	Vcl::Graphics::Runtime::PipelineStateDescription pipeline_desc;
	pipeline_desc.InputAssembly.PrimitiveRestartEnable = false;
	pipeline_desc.InputAssembly.Topology = Vcl::Graphics::Runtime::PrimitiveType::Pointlist;
	pipeline_desc.InputLayout = opaqueTetraLayout;

	size_t vs_code_size = 0;
	const char* vs_code = readBinaryFile("triangle.vert.spv", &vs_code_size);
	Vcl::Graphics::Runtime::Vulkan::Shader vs{ *context, Vcl::Graphics::Runtime::ShaderType::VertexShader, 0, vs_code, vs_code_size };
	pipeline_desc.VertexShader = &vs;

	PipelineState state{ context.get(), &layout, render_pass, pipeline_desc };

	// Build command buffers
	std::vector<Vcl::Graphics::Vulkan::CommandBuffer> cmds;
	buildCommandBuffers(&framebuffer, context->commandPool(0, CommandBufferType::Default), render_pass, 1280, 720, cmds);
	*/
	// End: Scene setup

	return 0;

	// Enter the event-loop
	CommandBuffer post_present{ *context, context->commandPool(0, CommandBufferType::Default) };
	Semaphore presentComplete{ context.get() };
	Semaphore renderComplete{ context.get() };
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		VkResult err;

		// Render the scene
		uint32_t curr_buf;
		err = surface->swapChain()->acquireNextImage(presentComplete, &curr_buf);
		if (err != VK_SUCCESS)
			continue;
		
		// Put post present barrier into command buffer
		post_present.begin();
		post_present.returnFromPresent(surface->swapChain()->image(curr_buf));
		post_present.end();

		// Submit to the queue
		queue.submit(post_present);
		queue.waitIdle();

		// Submit to the graphics queue
		//VkPipelineStageFlags pipelineStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		//VkSemaphore s0 = presentComplete;
		//VkSemaphore s1 = renderComplete;
		//VkCommandBuffer b0 = cmds[curr_buf];
		//queue.submit({ &b0, 1 }, &pipelineStages, { &s0, 1 }, { &s1, 1 });

		// Present the current buffer to the swap chain
		// We pass the signal semaphore from the submit info
		// to ensure that the image is not rendered until
		// all commands have been submitted
		surface->swapChain()->queuePresent(queue, curr_buf, renderComplete);
	}

	return 0;
}
