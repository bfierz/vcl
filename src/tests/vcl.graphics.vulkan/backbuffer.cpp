/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <vcl/graphics/vulkan/context.h>
#include <vcl/graphics/vulkan/device.h>
#include <vcl/graphics/vulkan/platform.h>
#include <vcl/graphics/vulkan/swapchain.h>

// Windows API
#ifdef VCL_ABI_WINAPI
#	include <windows.h>
#endif

// Test
#include "test.h"

#ifdef VCL_ABI_WINAPI
static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

using VkWin32SurfaceCreateFlagsKHR = VkFlags;
struct VkWin32SurfaceCreateInfoKHR
{
	VkStructureType                 sType;
	const void*                     pNext;
	VkWin32SurfaceCreateFlagsKHR    flags;
	HINSTANCE                       hinstance;
	HWND                            hwnd;
};

typedef VkResult(APIENTRY *PFN_vkCreateWin32SurfaceKHR)(VkInstance, const VkWin32SurfaceCreateInfoKHR*, const VkAllocationCallbacks*, VkSurfaceKHR*);
typedef VkBool32(APIENTRY *PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR)(VkPhysicalDevice, uint32_t);

static VkResult vkWin32CreateWindowSurface
(
	VkInstance instance,
	HWND window,
	const VkAllocationCallbacks* allocator,
	VkSurfaceKHR* surface
)
{
	VkResult err;
	VkWin32SurfaceCreateInfoKHR sci;
	PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR;

	vkCreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)
		vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR");
	//if (!vkCreateWin32SurfaceKHR)
	//{
	//	_glfwInputError(GLFW_API_UNAVAILABLE,
	//		"Win32: Vulkan instance missing VK_KHR_win32_surface extension");
	//	return VK_ERROR_EXTENSION_NOT_PRESENT;
	//}

	memset(&sci, 0, sizeof(sci));
	sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	sci.hinstance = GetModuleHandle(NULL);
	sci.hwnd = window;

	err = vkCreateWin32SurfaceKHR(instance, &sci, allocator, surface);
	//if (err)
	//{
	//	_glfwInputError(GLFW_PLATFORM_ERROR,
	//		"Win32: Failed to create Vulkan surface: %s",
	//		_glfwGetVulkanResultString(err));
	//}

	return err;
}
#endif

class VulkanBackbufferTest : public VulkanTest
{
public:
	void SetUp() override
	{
		_context_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
		_platform_extensions =
		{
			"VK_KHR_surface",
#ifdef VCL_ABI_WINAPI
			"VK_KHR_win32_surface",
#endif
		};

		VulkanTest::SetUp();

#ifdef VCL_ABI_WINAPI
		WNDCLASS wc = { 0 };
		wc.lpfnWndProc = WndProc;
		wc.hInstance = GetModuleHandle(NULL);
		wc.hbrBackground = (HBRUSH)(COLOR_BACKGROUND);
		wc.lpszClassName = "VulkanContextWindowClass";
		wc.style = CS_OWNDC;
		RegisterClass(&wc);
		_window_handle = CreateWindowEx(0, wc.lpszClassName, "VulkanContextWindow", 0, 0, 0, 512, 512, HWND_MESSAGE, 0, 0, 0);

		// Create a WSI surface for the window
		vkWin32CreateWindowSurface(*_platform, _window_handle, nullptr, &_surface_ctx);
#endif
	}

	void TearDown() override
	{
#ifdef VCL_ABI_WINAPI
		CloseWindow(_window_handle);
#endif
		VulkanTest::TearDown();
	}

protected:
#ifdef VCL_ABI_WINAPI
	//! Native window handle of the test window
	HWND _window_handle;

	//! Vulkan surface context
	VkSurfaceKHR _surface_ctx;
#endif
};

TEST_F(VulkanBackbufferTest, Create)
{
	using namespace Vcl::Graphics::Vulkan;
	using namespace Vcl::Graphics;

	// Command queue used for back-buffer initialization
	CommandQueue queue{ _context.get(), VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT, 0 };

	BasicSurfaceDescription desc;
	desc.Surface = _surface_ctx;
	desc.NumberOfImages = 4;
	desc.ColourFormat = VK_FORMAT_B8G8R8A8_UNORM;
	desc.DepthFormat = VK_FORMAT_D32_SFLOAT;
	auto surface = createBasicSurface(*_platform, *_context, queue, desc);

	// Iterate through all images in the swap chain and present them
	CommandBuffer post_present{ *_context, _context->commandPool(0, CommandBufferType::Default) };
	for (int i = 0; i < 4; i++)
	{
		// Render the scene
		uint32_t curr_buf;
		Semaphore present_complete{ _context.get() };
		VkResult err = surface->swapChain()->acquireNextImage(present_complete, &curr_buf);
		if (err != VK_SUCCESS)
			continue;

		// Put post present barrier into command buffer
		post_present.begin();
		post_present.returnFromPresent(surface->swapChain()->image(curr_buf));
		post_present.end();

		// Submit to the queue
		queue.submit(post_present, VK_NULL_HANDLE);
		queue.waitIdle();

		// Present the current buffer to the swap chain
		// We pass the signal semaphore from the submit info
		// to ensure that the image is not rendered until
		// all commands have been submitted
		surface->swapChain()->queuePresent(queue, curr_buf, VK_NULL_HANDLE);
	}

}
