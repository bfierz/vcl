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

// C++ standard library
#include <cstring>
#include <stdexcept>

// Windows Graphics
#ifdef _WIN32
#	include <dxgi1_6.h>
#	ifdef VCL_DEBUG
#		include <dxgidebug.h>
#	endif
#endif

// VCL
#include <vcl/core/contract.h>
#include <directx/d3dx12.h>
#include <vcl/graphics/d3d12/d3d.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	using namespace Microsoft::WRL;

	namespace {
		constexpr D3D_FEATURE_LEVEL MinimumFeatureLevel = D3D_FEATURE_LEVEL_12_0;

#ifdef VCL_DEBUG
		bool isDebuggerAttached()
		{
#	ifdef _WIN32
			return IsDebuggerPresent() != FALSE;
#	else
			return false;
#	endif
		}

		const char* severityName(D3D12_MESSAGE_SEVERITY severity)
		{
			switch (severity)
			{
			case D3D12_MESSAGE_SEVERITY_CORRUPTION:
				return "corruption";
			case D3D12_MESSAGE_SEVERITY_ERROR:
				return "error";
			case D3D12_MESSAGE_SEVERITY_WARNING:
				return "warning";
			case D3D12_MESSAGE_SEVERITY_INFO:
				return "info";
			case D3D12_MESSAGE_SEVERITY_MESSAGE:
				return "message";
			}
			return "unknown";
		}

		//! Print debug-layer messages when no debugger can break on them
		void __stdcall printDebugMessage(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR description, void*)
		{
			if (severity == D3D12_MESSAGE_SEVERITY_INFO || severity == D3D12_MESSAGE_SEVERITY_MESSAGE)
				return;

			std::cerr << "D3D12 " << severityName(severity) << " #" << static_cast<int>(id) << ": " << description << std::endl;
		}
#endif

		struct Adapter
		{
			ComPtr<IUnknown> handle;
			AdapterInfo info;
		};

		std::vector<AdapterInfo> infosOf(const std::vector<Adapter>& adapters)
		{
			std::vector<AdapterInfo> infos;
			infos.reserve(adapters.size());
			for (const auto& adapter : adapters)
				infos.push_back(adapter.info);

			return infos;
		}

		bool supportsMinimumFeatureLevel(IUnknown* adapter)
		{
			return SUCCEEDED(D3D12CreateDevice(adapter, MinimumFeatureLevel, __uuidof(ID3D12Device), nullptr));
		}

#ifdef _WIN32
		std::string toUtf8(const wchar_t* str)
		{
			const int size = WideCharToMultiByte(CP_UTF8, 0, str, -1, nullptr, 0, nullptr, nullptr);
			if (size <= 1)
				return {};

			std::string result(size, '\0');
			WideCharToMultiByte(CP_UTF8, 0, str, -1, &result[0], size, nullptr, nullptr);
			result.resize(size - 1);
			return result;
		}

		void addAdapter(IDXGIAdapter1* adapter, bool is_hardware, std::vector<Adapter>& adapters)
		{
			if (!supportsMinimumFeatureLevel(adapter))
				return;

			DXGI_ADAPTER_DESC1 desc;
			if (FAILED(adapter->GetDesc1(&desc)))
				return;

			adapters.push_back({ adapter, { toUtf8(desc.Description), is_hardware, desc.DedicatedVideoMemory } });
		}

		std::vector<Adapter> queryAdapters()
		{
			ComPtr<IDXGIFactory4> factory;
#	ifdef VCL_DEBUG
			// The DXGI debug layer is optional (Windows "Graphics Tools" feature)
			if (FAILED(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory))))
#	endif
				if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))))
					return {};

			std::vector<Adapter> adapters;
			ComPtr<IDXGIAdapter1> adapter;
			for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
			{
				// Software adapters are covered by the WARP adapter below
				DXGI_ADAPTER_DESC1 desc;
				if (SUCCEEDED(adapter->GetDesc1(&desc)) && (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0)
					addAdapter(adapter.Get(), true, adapters);
			}

			if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))))
				addAdapter(adapter.Get(), false, adapters);

			return adapters;
		}
#else
		template<typename T>
		bool queryProperty(IDXCoreAdapter* adapter, DXCoreAdapterProperty property, T* value)
		{
			return adapter->IsPropertySupported(property) &&
				   SUCCEEDED(adapter->GetProperty(property, sizeof(T), value));
		}

		std::string queryName(IDXCoreAdapter* adapter)
		{
			const auto property = DXCoreAdapterProperty::DriverDescription;
			size_t size = 0;
			if (!adapter->IsPropertySupported(property) || FAILED(adapter->GetPropertySize(property, &size)) || size == 0)
				return {};

			std::string name(size, '\0');
			if (FAILED(adapter->GetProperty(property, size, &name[0])))
				return {};

			name.resize(std::strlen(name.c_str()));
			return name;
		}

		std::vector<Adapter> queryAdapters()
		{
			ComPtr<IDXCoreAdapterFactory> factory;
			if (FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory))))
				return {};

			const GUID attributes[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS };
			ComPtr<IDXCoreAdapterList> list;
			if (FAILED(factory->CreateAdapterList(1, attributes, IID_PPV_ARGS(&list))))
				return {};

			std::vector<Adapter> adapters;
			for (uint32_t i = 0; i < list->GetAdapterCount(); ++i)
			{
				ComPtr<IDXCoreAdapter> adapter;
				if (FAILED(list->GetAdapter(i, IID_PPV_ARGS(&adapter))) || !supportsMinimumFeatureLevel(adapter.Get()))
					continue;

				AdapterInfo info;
				info.name = queryName(adapter.Get());
				queryProperty(adapter.Get(), DXCoreAdapterProperty::IsHardware, &info.isHardware);
				queryProperty(adapter.Get(), DXCoreAdapterProperty::DedicatedAdapterMemory, &info.dedicatedVideoMemory);
				adapters.push_back({ adapter, std::move(info) });
			}

			return adapters;
		}
#endif
	}

#ifdef _WIN32
	bool isRuntimeAvailable()
	{
		return true;
	}
#endif

	std::vector<AdapterInfo> enumerateAdapters()
	{
		return infosOf(queryAdapters());
	}

	int selectAdapter(const std::vector<AdapterInfo>& adapters, DeviceType type)
	{
		int selected = -1;
		for (int i = 0; i < static_cast<int>(adapters.size()); ++i)
		{
			const auto& adapter = adapters[i];
			if (type == DeviceType::Warp)
			{
				if (!adapter.isHardware)
					return i;
			} else if (adapter.isHardware && (selected < 0 || adapter.dedicatedVideoMemory > adapters[selected].dedicatedVideoMemory))
			{
				selected = i;
			}
		}

		return selected;
	}

	Device::Device(DeviceType type)
	{
		using namespace Microsoft::WRL;

		enableDebugLayer();

		const auto adapters = queryAdapters();
		const int selected = selectAdapter(infosOf(adapters), type);
		if (selected < 0)
			throw std::runtime_error(type == DeviceType::Warp ? "No WARP adapter found" : "No D3D12 hardware adapter found");

		_adapter = adapters[selected].info;
		if (FAILED(D3D12CreateDevice(adapters[selected].handle.Get(), MinimumFeatureLevel, IID_PPV_ARGS(&_d3dDevice))) || !_d3dDevice)
			throw std::runtime_error("Failed to create the D3D12 device on " + _adapter.name);

		// Query raytracing capabilities (ID3D12Device5 is optional)
		ComPtr<ID3D12Device5> device5;
		if (SUCCEEDED(_d3dDevice.As(&device5)))
		{
			D3D12_FEATURE_DATA_D3D12_OPTIONS5 caps = {};
			if (SUCCEEDED(device5->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &caps, sizeof(caps))))
			{
				_raytracingTier = caps.RaytracingTier;
			}
		}

		queryFeatures();

		// Enable debug messages in debug mode
#ifdef VCL_DEBUG
		ComPtr<ID3D12InfoQueue> info_queue;
		if (SUCCEEDED(_d3dDevice.As(&info_queue)))
		{
			// Breaking without a debugger raises an unhandled exception, which
			// blocks non-interactive runs (e.g. CI) on a crash dialog
			if (isDebuggerAttached())
			{
				info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
				info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
				info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);
			} else
			{
				// Message callbacks need Windows 11 or the Agility SDK
				ComPtr<ID3D12InfoQueue1> info_queue1;
				DWORD cookie = 0;
				if (FAILED(info_queue.As(&info_queue1)) || FAILED(info_queue1->RegisterMessageCallback(&printDebugMessage, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie)))
					std::cerr << "D3D12 debug messages are not printed: the debug layer does not support message callbacks" << std::endl;
			}

			// Suppress whole categories of messages
			//D3D12_MESSAGE_CATEGORY Categories[] = {};

			// Suppress messages based on their severity level
			D3D12_MESSAGE_SEVERITY Severities[] = {
				D3D12_MESSAGE_SEVERITY_INFO
			};

			// Suppress individual messages by their ID
			//D3D12_MESSAGE_ID DenyIds[] = {
			//	D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,   // I'm really not sure how to avoid this message.
			//	D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,                         // This warning occurs when using capture frame while graphics debugging.
			//	D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,                       // This warning occurs when using capture frame while graphics debugging.
			//};

			D3D12_INFO_QUEUE_FILTER NewFilter = {};
			//NewFilter.DenyList.NumCategories = _countof(Categories);
			//NewFilter.DenyList.pCategoryList = Categories;
			NewFilter.DenyList.NumSeverities = _countof(Severities);
			NewFilter.DenyList.pSeverityList = Severities;
			//NewFilter.DenyList.NumIDs = _countof(DenyIds);
			//NewFilter.DenyList.pIDList = DenyIds;

			VCL_DIRECT3D_SAFE_CALL(info_queue->PushStorageFilter(&NewFilter));
		}
#endif

		_defaultQueue = std::make_unique<CommandQueue>(this);
	}

	Device::~Device()
	{
#if defined(_WIN32) && defined(VCL_DEBUG)
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
		// The debug layer is optional (Windows "Graphics Tools" feature; not
		// provided by vkd3d-proton)
		ComPtr<ID3D12Debug> debug_interface;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_interface))))
			debug_interface->EnableDebugLayer();
		else
			std::cout << "D3D12 debug layer is not available" << std::endl;
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

	void Device::queryFeatures()
	{
		// The runtime rejects shader models it doesn't know, so query from the newest down
		const D3D_SHADER_MODEL models[] = {
			D3D_SHADER_MODEL_6_9, D3D_SHADER_MODEL_6_8, D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_6,
			D3D_SHADER_MODEL_6_5, D3D_SHADER_MODEL_6_4, D3D_SHADER_MODEL_6_3, D3D_SHADER_MODEL_6_2,
			D3D_SHADER_MODEL_6_1, D3D_SHADER_MODEL_6_0
		};
		for (auto model : models)
		{
			D3D12_FEATURE_DATA_SHADER_MODEL data = { model };
			if (SUCCEEDED(_d3dDevice->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &data, sizeof(data))))
			{
				_shaderModel = data.HighestShaderModel;
				break;
			}
		}

		D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
		if (SUCCEEDED(_d3dDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))))
			_resourceBindingTier = options.ResourceBindingTier;

		// Heap indexing (ResourceDescriptorHeap, SamplerDescriptorHeap) needs
		// shader model 6.6 and resource binding tier 3
		_features.BindlessResources = _shaderModel >= D3D_SHADER_MODEL_6_6 && _resourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_3;
		_features.GeometryShaders = true;
		_features.TessellationShaders = true;
		// The backend has no mesh pipeline
		_features.MeshShaders = false;
		_features.TaskShaders = false;
		_features.IndependentBlend = true;
		_features.DualSourceBlend = true;
		_features.LogicOp = options.OutputMergerLogicOp == TRUE;
		_features.WireframeFill = true;
		_features.DepthBiasClamp = true;
		_features.DepthClipControl = true;
		_features.SamplerAnisotropy = true;
		_features.MaxRenderTargets = D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT;
		// Root signatures hold 64 DWORDs; the bindless layout only uses root constants
		_features.MaxPushConstantSize = 64 * sizeof(uint32_t);
		_features.MaxResourceHeapCapacity = D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2;
		_features.MaxSamplerHeapCapacity = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
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
