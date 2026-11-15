/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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
#include <vcl/config/eigen.h>
#include <vcl/config/direct3d12.h>

// C++ Standard library
#include <cstdint>
#include <string>
#include <vector>

// Windows Runtime Library
#include <wrl/client.h>

// VCL
#include <vcl/graphics/d3d12/commandqueue.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	enum class DeviceType
	{
		Hardware,
		Warp
	};

	//! Description of a graphics adapter
	struct AdapterInfo
	{
		//! Name reported by the driver (UTF-8)
		std::string name;

		//! False for software adapters such as WARP
		bool isHardware{ false };

		//! Memory dedicated to the adapter in bytes
		uint64_t dedicatedVideoMemory{ 0 };
	};

	//! \returns true if the D3D12 runtime can be loaded. It is always available on
	//!          Windows. On Linux, WSL provides it (libd3d12.so and libdxcore.so).
	bool isRuntimeAvailable();

	//! Lists the adapters on which a device with feature level 12_0 can be created.
	//! Uses DXGI on Windows and DXCore on Linux.
	std::vector<AdapterInfo> enumerateAdapters();

	//! Chooses the adapter for a device of the given type: the hardware adapter with
	//! the most dedicated memory, or the first software adapter for WARP.
	//! \returns the index into 'adapters', or -1 if none fits
	int selectAdapter(const std::vector<AdapterInfo>& adapters, DeviceType type);

	class Device final
	{
	public:
		template<typename T>
		using ComPtr = Microsoft::WRL::ComPtr<T>;

		//! Creates a device on the adapter chosen by 'selectAdapter'
		//! \throws std::runtime_error if no adapter fits or the device cannot be created
		Device(DeviceType type);
		~Device();

		const AdapterInfo& adapter() const { return _adapter; }
		const std::string& adapterName() const { return _adapter.name; }

		ID3D12Device* nativeDevice() const { return _d3dDevice.Get(); }

		CommandQueue* defaultQueue() const { return _defaultQueue.get(); }

		//! \name Query feature support
		//! \{
		//! \returns the D3D12 raytracing support level
		D3D12_RAYTRACING_TIER raytracingSupport() const { return _raytracingTier; }
		//! \}

		//! \name D3D12 Helpers
		//! \{
		ComPtr<ID3D12Fence> createFence();

		//! Create a new command queue
		ComPtr<ID3D12CommandQueue> createCommandQueue(D3D12_COMMAND_LIST_TYPE type);

		ComPtr<ID3D12CommandAllocator> createCommandAllocator(D3D12_COMMAND_LIST_TYPE type);

		ComPtr<ID3D12GraphicsCommandList4> createCommandList(ID3D12CommandAllocator* cmd_allocator, D3D12_COMMAND_LIST_TYPE type);

		ComPtr<ID3D12RootSignature> createRootSignature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc);

		ComPtr<ID3D12DescriptorHeap> createDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t numDescriptors, bool visible_to_shaders);

		ComPtr<ID3D12PipelineState> createComputePipelineState(const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc);

		ComPtr<ID3D12PipelineState> createGraphicsPipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc);
		//! \}

	private:
		void enableDebugLayer();

		ComPtr<ID3D12Device> _d3dDevice;

		AdapterInfo _adapter;

		std::unique_ptr<CommandQueue> _defaultQueue;

		//! \name Feature Support
		//! \{
		D3D12_RAYTRACING_TIER _raytracingTier{ D3D12_RAYTRACING_TIER_NOT_SUPPORTED };
		//! \}
	};
}}}
