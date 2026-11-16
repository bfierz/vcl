/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2026 Basil Fierz
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

// Linux has no import libraries for D3D12: WSL provides the runtime as shared
// libraries mounted from the Windows driver store (/usr/lib/wsl/lib). This file
// defines the D3D12 and DXCore entry points declared by the headers and forwards
// them to those libraries, which are loaded on first use. Without them, the
// entry points fail with an error instead of the program failing to start.

#include <vcl/graphics/d3d12/device.h>

// C runtime
#include <dlfcn.h>

namespace {
	//! HRESULT_FROM_WIN32(ERROR_MOD_NOT_FOUND)
	constexpr HRESULT ModuleNotFound = static_cast<HRESULT>(0x8007007EL);

	void* openLibrary(const char* name)
	{
		return dlopen(name, RTLD_NOW | RTLD_LOCAL);
	}

	void* d3d12Library()
	{
		static void* const library = openLibrary("libd3d12.so");
		return library;
	}

	void* dxcoreLibrary()
	{
		static void* const library = openLibrary("libdxcore.so");
		return library;
	}

	//! Looks up 'name' in 'library'. dlsym with a library handle only searches
	//! that library, so this never resolves to the forwarding functions below.
	template<typename Function>
	Function lookup(void* library, const char* name)
	{
		return library ? reinterpret_cast<Function>(dlsym(library, name)) : nullptr;
	}
}

namespace Vcl { namespace Graphics { namespace D3D12 {
	bool isRuntimeAvailable()
	{
		return d3d12Library() && dxcoreLibrary();
	}
}}}

extern "C"
{
	HRESULT WINAPI D3D12CreateDevice(IUnknown* adapter, D3D_FEATURE_LEVEL minimum_feature_level, REFIID riid, void** device)
	{
		static const auto function = lookup<PFN_D3D12_CREATE_DEVICE>(d3d12Library(), "D3D12CreateDevice");
		return function ? function(adapter, minimum_feature_level, riid, device) : ModuleNotFound;
	}

	HRESULT WINAPI D3D12GetDebugInterface(REFIID riid, void** debug)
	{
		static const auto function = lookup<PFN_D3D12_GET_DEBUG_INTERFACE>(d3d12Library(), "D3D12GetDebugInterface");
		return function ? function(riid, debug) : ModuleNotFound;
	}

	HRESULT WINAPI D3D12SerializeRootSignature(const D3D12_ROOT_SIGNATURE_DESC* desc, D3D_ROOT_SIGNATURE_VERSION version, ID3DBlob** blob, ID3DBlob** error_blob)
	{
		static const auto function = lookup<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(d3d12Library(), "D3D12SerializeRootSignature");
		return function ? function(desc, version, blob, error_blob) : ModuleNotFound;
	}

	HRESULT WINAPI D3D12SerializeVersionedRootSignature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc, ID3DBlob** blob, ID3DBlob** error_blob)
	{
		static const auto function = lookup<PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE>(d3d12Library(), "D3D12SerializeVersionedRootSignature");
		return function ? function(desc, blob, error_blob) : ModuleNotFound;
	}

	HRESULT STDAPICALLTYPE DXCoreCreateAdapterFactory(REFIID riid, void** factory)
	{
		using PFN_DXCORE_CREATE_ADAPTER_FACTORY = HRESULT(STDAPICALLTYPE*)(REFIID, void**);
		static const auto function = lookup<PFN_DXCORE_CREATE_ADAPTER_FACTORY>(dxcoreLibrary(), "DXCoreCreateAdapterFactory");
		return function ? function(riid, factory) : ModuleNotFound;
	}
}
