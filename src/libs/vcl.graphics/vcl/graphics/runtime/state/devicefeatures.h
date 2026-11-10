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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <cstdint>
#include <string>

// VCL
#include <vcl/graphics/runtime/state/pipelinestate.h>
#include <vcl/graphics/runtime/state/resourcetable.h>

namespace Vcl { namespace Graphics { namespace Runtime {
	//! Optional capabilities of a device.
	//! Each backend reports what the shared descriptions may use.
	struct DeviceFeatures
	{
		//! Bindless resource heap (Vulkan / D3D12)
		bool BindlessResources{ false };

		bool GeometryShaders{ false };
		bool TessellationShaders{ false };
		bool MeshShaders{ false };
		bool TaskShaders{ false };

		//! Different blend states per render-target
		bool IndependentBlend{ false };

		//! Blend factors using the second colour output of the fragment shader
		bool DualSourceBlend{ false };

		bool LogicOp{ false };
		bool WireframeFill{ false };
		bool DepthBiasClamp{ false };

		//! Depth clipping can be disabled
		bool DepthClipControl{ false };

		bool SamplerAnisotropy{ false };

		//! Number of render-targets a pipeline can render to
		uint32_t MaxRenderTargets{ 0 };

		//! Size of the push-constant / root-constant block in bytes
		uint32_t MaxPushConstantSize{ 0 };

		//! Capacity limits of the bindless heaps
		uint32_t MaxResourceHeapCapacity{ 0 };
		uint32_t MaxSamplerHeapCapacity{ 0 };
	};

	//! Check a description against the features of a device
	//! \returns An empty string if the description is supported, otherwise
	//!          the description of the first unsupported property
	std::string validate(const PipelineStateDescription& desc, const DeviceFeatures& features);
	std::string validate(const MeshPipelineStateDescription& desc, const DeviceFeatures& features);
	std::string validate(const BindlessLayoutDescription& desc, const DeviceFeatures& features);
}}}
