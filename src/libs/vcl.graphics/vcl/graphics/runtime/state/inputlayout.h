/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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
#include <algorithm>
#include <vector>

// VCL
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace Runtime
{
	template<typename VertexDataType>
	struct InputLayoutTypeTrait
	{
		typedef VertexDataType Type;
		static const SurfaceFormat Format;
	};
	
	enum class VertexDataClassification
	{
		VertexDataPerObject,
		VertexDataPerInstance
	};

	struct InputLayoutElement
	{
		std::string Name;
		SurfaceFormat Format;
		unsigned int NumberLocations;
		unsigned int Offset;
		VertexDataClassification StreamType;
		unsigned int StepRate;
	};

	class InputLayoutDescription
	{
	public:
		InputLayoutDescription() = default;

	public:
		InputLayoutDescription(std::initializer_list<InputLayoutElement> init)
		: _elements(init)
		{
			_locations.reserve(_elements.size());

			int loc = 0;
			for (const auto& elem : _elements)
			{
				_locations.emplace_back(loc);
				loc += std::max(1, (int) elem.NumberLocations);
			}
		}
		InputLayoutDescription(const InputLayoutDescription& rhs)
		{
			_elements  = rhs._elements;
			_locations = rhs._locations;
		}
		InputLayoutDescription(InputLayoutDescription&& rhs) 
		{
			std::swap(_elements, rhs._elements);
			std::swap(_locations, rhs._locations);
		}

	public:
		std::vector<InputLayoutElement>::iterator begin() { return _elements.begin(); }
		std::vector<InputLayoutElement>::const_iterator begin() const { return _elements.cbegin(); }
		std::vector<InputLayoutElement>::const_iterator cbegin() const { return _elements.cbegin(); }

		std::vector<InputLayoutElement>::iterator end() { return _elements.end(); }
		std::vector<InputLayoutElement>::const_iterator end() const { return _elements.cend(); }
		std::vector<InputLayoutElement>::const_iterator cend() const { return _elements.cend(); }

		std::vector<InputLayoutElement>::size_type size() const { return _elements.size(); }

		const InputLayoutElement& operator[] (size_t idx) const { return _elements[idx]; }

		int location(size_t idx) const { return _locations[idx]; }

	private:
		std::vector<InputLayoutElement> _elements;
		std::vector<int> _locations;
	};
}}}
