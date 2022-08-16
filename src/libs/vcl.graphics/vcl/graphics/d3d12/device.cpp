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
#include <vcl/graphics/d3d12/device.h>

// Windows Graphics
#include <dxgi1_6.h>
#ifdef VCL_DEBUG
#	include <dxgidebug.h>
#endif

// VCL
#include <vcl/core/contract.h>
#include <vcl/graphics/d3d12/3rdparty/d3dx12.h>
#include <vcl/graphics/d3d12/d3d.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	using namespace Microsoft::WRL;

	namespace {
		ComPtr<IDXGIFactory4> queryDXGIFactory()
		{
			ComPtr<IDXGIFactory4> factory;
			UINT flags = 0;
#ifdef VCL_DEBUG
			flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
			VCL_DIRECT3D_SAFE_CALL(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory)));
			return factory;
		}

		ComPtr<IDXGIAdapter4> selectAdapter(bool force_warp_device, D3D_FEATURE_LEVEL min_feature_level)
		{
			auto factory = queryDXGIFactory();

			ComPtr<IDXGIAdapter1> dxgi_adapter1;
			ComPtr<IDXGIAdapter4> dxgi_adapter4;

			if (force_warp_device)
			{
				VCL_DIRECT3D_SAFE_CALL(factory->EnumWarpAdapter(IID_PPV_ARGS(&dxgi_adapter1)));
				VCL_DIRECT3D_SAFE_CALL(dxgi_adapter1.As(&dxgi_adapter4));
			} else
			{
				SIZE_T video_memory = 0;
				for (UINT i = 0; factory->EnumAdapters1(i, &dxgi_adapter1) != DXGI_ERROR_NOT_FOUND; ++i)
				{
					DXGI_ADAPTER_DESC1 adapter_desc;
					dxgi_adapter1->GetDesc1(&adapter_desc);

					// Ignore software devices
					if (adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
						continue;

					// Ignore any devie that cannot create a specific D3D feature level
					if (FAILED(D3D12CreateDevice(dxgi_adapter1.Get(), min_feature_level, __uuidof(ID3D12Device), nullptr)))
						continue;

					// Use the video memory as selection criterion
					if (adapter_desc.DedicatedVideoMemory > video_memory)
					{
						video_memory = adapter_desc.DedicatedVideoMemory;
						VCL_DIRECT3D_SAFE_CALL(dxgi_adapter1.As(&dxgi_adapter4));
					}
				}
			}

			return dxgi_adapter4;
		}
	}

	Device::Device(DeviceType type)
	{
		using namespace Microsoft::WRL;

		enableDebugLayer();

		auto adapter = selectAdapter(type == DeviceType::Warp, D3D_FEATURE_LEVEL_12_0);
		HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&_d3dDevice));
		VclEnsure(SUCCEEDED(hr) && _d3dDevice, "Native device is allocated.");

		// Query raytracing capabilities
		ComPtr<ID3D12Device5> device5;
		VCL_DIRECT3D_SAFE_CALL(_d3dDevice.As(&device5));
		if (device5)
		{
			D3D12_FEATURE_DATA_D3D12_OPTIONS5 caps = {};
			if (SUCCEEDED(device5->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &caps, sizeof(caps))))
			{
				_raytracingTier = caps.RaytracingTier;
			}
		}

		// Store adapter information
		DXGI_ADAPTER_DESC3 adapter_desc;
		adapter->GetDesc3(&adapter_desc);
		_adapterName.assign(adapter_desc.Description);

		// Enable debug messages in debug mode
#ifdef VCL_DEBUG
		ComPtr<ID3D12InfoQueue> info_queue;
		if (SUCCEEDED(_d3dDevice.As(&info_queue)))
		{
			info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
			info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
			info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);

			// Suppress whole categories of messages
			//D3D12_MESSAGE_CATEGORY Categories[] = {};

			// Suppress messages based on their severity level
			D3D12_MESSAGE_SEVERITY Severities[] = {
				D3D12_MESSAGE_SEVERITY_INFO
			};

			// Suppress individual messages by their ID
			// Some recommendations: https://stackoverflow.com/a/69833651
			D3D12_MESSAGE_ID DenyIds[] = {
				D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,                             // This warning occurs when using capture frame while graphics debugging.
				D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,                           // This warning occurs when using capture frame while graphics debugging.
				D3D12_MESSAGE_ID_RESOURCE_BARRIER_MISMATCHING_COMMAND_LIST_TYPE,    // Windows 11 related debug layer issue
				D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_WRONGSWAPCHAINBUFFERREFERENCE, // Workaround for debug layer issues on hybrid-graphics systems
			};

			D3D12_INFO_QUEUE_FILTER filter = {};
			//filter.DenyList.NumCategories = _countof(Categories);
			//filter.DenyList.pCategoryList = Categories;
			filter.DenyList.NumSeverities = _countof(Severities);
			filter.DenyList.pSeverityList = Severities;
			filter.DenyList.NumIDs = _countof(DenyIds);
			filter.DenyList.pIDList = DenyIds;

			VCL_DIRECT3D_SAFE_CALL(info_queue->PushStorageFilter(&filter));
		}
#endif

		_defaultQueue = std::make_unique<CommandQueue>(this);
	}

	Device::~Device()
	{
#ifdef VCL_DEBUG
		ComPtr<IDXGIDebug1> debug_interface;
		if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&debug_interface))))
		{
			//debug_interface->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_SUMMARY);
		}
#endif
	}

	void Device::enableDebugLayer()
	{
#ifdef VCL_DEBUG
		ComPtr<ID3D12Debug> debug_interface;
		VCL_DIRECT3D_SAFE_CALL(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_interface)));
		debug_interface->EnableDebugLayer();
#endif
	}

	ComPtr<ID3D12Fence> Device::createFence()
	{
		ComPtr<ID3D12Fence> fence;
		VCL_DIRECT3D_SAFE_CALL(_d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
		return fence;
	}

	ComPtr<ID3D12CommandQueue> Device::createCommandQueue(D3D12_COMMAND_LIST_TYPE type)
	{
		ComPtr<ID3D12CommandQueue> cmd_queue;

		D3D12_COMMAND_QUEUE_DESC desc = {};
		desc.Type = type;
		desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
		desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		desc.NodeMask = 0;

		VCL_DIRECT3D_SAFE_CALL(_d3dDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&cmd_queue)));

		return cmd_queue;
	}

	ComPtr<ID3D12CommandAllocator> Device::createCommandAllocator(D3D12_COMMAND_LIST_TYPE type)
	{
		ComPtr<ID3D12CommandAllocator> cmd_allocator;
		VCL_DIRECT3D_SAFE_CALL(_d3dDevice->CreateCommandAllocator(type, IID_PPV_ARGS(&cmd_allocator)));

		return cmd_allocator;
	}

	ComPtr<ID3D12GraphicsCommandList4> Device::createCommandList(ID3D12CommandAllocator* cmd_allocator, D3D12_COMMAND_LIST_TYPE type)
	{
		ComPtr<ID3D12GraphicsCommandList4> cmd_list;
		VCL_DIRECT3D_SAFE_CALL(_d3dDevice->CreateCommandList(0, type, cmd_allocator, nullptr, IID_PPV_ARGS(&cmd_list)));

		return cmd_list;
	}

	ComPtr<ID3D12RootSignature> Device::createRootSignature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc)
	{
		ComPtr<ID3DBlob> signature_blob;
		ComPtr<ID3DBlob> error;
		ComPtr<ID3D12RootSignature> signature;
		D3DX12SerializeVersionedRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature_blob, &error);
		_d3dDevice->CreateRootSignature(0, signature_blob->GetBufferPointer(), signature_blob->GetBufferSize(), IID_PPV_ARGS(&signature));

		return signature;
	}

	ComPtr<ID3D12DescriptorHeap> Device::createDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t numDescriptors, bool visible_to_shaders)
	{
		ComPtr<ID3D12DescriptorHeap> heap;

		D3D12_DESCRIPTOR_HEAP_DESC desc = {};
		desc.NumDescriptors = numDescriptors;
		desc.Type = type;
		desc.Flags = visible_to_shaders ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		desc.NodeMask = 0;

		VCL_DIRECT3D_SAFE_CALL(_d3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)));

		return heap;
	}

	ComPtr<ID3D12PipelineState> Device::createComputePipelineState(const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc)
	{
		ComPtr<ID3D12PipelineState> pso;
		VCL_DIRECT3D_SAFE_CALL(_d3dDevice->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)));

		return pso;
	}

	ComPtr<ID3D12PipelineState> Device::createGraphicsPipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc)
	{
		ComPtr<ID3D12PipelineState> pso;
		VCL_DIRECT3D_SAFE_CALL(_d3dDevice->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)));

		return pso;
	}
}}}
