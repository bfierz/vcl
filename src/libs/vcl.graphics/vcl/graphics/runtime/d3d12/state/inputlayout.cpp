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
#define VCL_GRAPHICS_HELIOS_OPENGL_INPUTLAYOUT_INST
#include <vcl/graphics/runtime/d3d12/state/inputlayout.h>

// VCL
#include <vcl/graphics/d3d12/d3d.h>

namespace Vcl { namespace Graphics { namespace Runtime { namespace D3D12
{
	std::vector<D3D12_INPUT_ELEMENT_DESC> toD3D12(const InputLayoutDescription& desc)
	{
		std::vector<D3D12_INPUT_ELEMENT_DESC> d3d12_desc;
		d3d12_desc.reserve(desc.attributes().size());

		int idx = 0;
		for (const auto& elem : desc.attributes())
		{
			const auto& binding = desc.binding(elem.InputSlot);
			for (int sub_loc = 0; sub_loc < std::max(1, (int)elem.NumberLocations); sub_loc++)
			{
				D3D12_INPUT_ELEMENT_DESC d3d12_elem;
				d3d12_elem.SemanticName = elem.Name.c_str();
				d3d12_elem.SemanticIndex = sub_loc;
				d3d12_elem.Format = Graphics::D3D12::D3D::toD3Denum(elem.Format);
				d3d12_elem.InputSlot = binding.Binding;
				d3d12_elem.AlignedByteOffset = elem.Offset;
				d3d12_elem.InputSlotClass = binding.InputRate == VertexDataClassification::VertexDataPerObject ? D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA : D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;
				d3d12_elem.InstanceDataStepRate = binding.InputRate == VertexDataClassification::VertexDataPerObject ? 0 : 1;
				d3d12_desc.emplace_back(d3d12_elem);
			}
		}

		return d3d12_desc;
	}
}}}}
