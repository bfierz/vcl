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
#include <vcl/graphics/imageprocessing/link.h>

// C++ Standard Library
#include <algorithm>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing {
	Slot::Slot(const std::string& id, Task* task)
	: _identifier(id)
	, _owner(task)
	{
	}

	InputSlot::InputSlot(const std::string& id, Task* task)
	: Slot(id, task)
	{
	}

	void InputSlot::disconnect()
	{
		if (_source)
		{
			auto iter = std::find(_source->_connections.begin(), _source->_connections.end(), this);
			VclCheck(iter != _source->_connections.end(), "Current connection is valid");
			if (iter != _source->_connections.end())
			{
				_source->_connections.erase(iter);
			}
		}
	}

	void InputSlot::setSource(OutputSlot* src)
	{
		if (_source)
			disconnect();

		_source = src;
		_resource = nullptr;

		_x = 0;
		_y = 0;
		_width = 0;
		_height = 0;

		if (src)
		{
			src->_connections.push_back(this);
		}
	}
	void InputSlot::setResource(const Runtime::Texture* input)
	{
		if (_source)
			disconnect();

		_source = nullptr;
		_resource = input;

		_x = 0;
		_y = 0;
		_width = (input) ? input->width() : 0;
		_height = (input) ? input->height() : 0;
	}
	void InputSlot::setResource(const Runtime::Texture* input, unsigned int width, unsigned int height)
	{
		VclRequire(implies(input, width <= (unsigned int)input->width()), "'x' + 'width' is in range.");
		VclRequire(implies(input, height <= (unsigned int)input->height()), "'y' + 'width' is in range.");

		if (_source)
			disconnect();

		_source = nullptr;
		_resource = input;

		_x = 0;
		_y = 0;
		_width = width;
		_height = height;
	}
	void InputSlot::setResource(const Runtime::Texture* input, unsigned int x, unsigned int y, unsigned int width, unsigned int height)
	{
		VclRequire(implies(input, x + width <= (unsigned int)input->width()), "'x' + 'width' is in range.");
		VclRequire(implies(input, y + height <= (unsigned int)input->height()), "'y' + 'width' is in range.");

		if (_source)
			disconnect();

		_source = nullptr;
		_resource = input;

		_x = x;
		_y = y;
		_width = width;
		_height = height;
	}

	const Runtime::Texture* InputSlot::resource() const
	{
		VclRequire(_resource || _source, "Slot is connected to a resource.");

		return _source ? _source->resource() : _resource;
	}
	unsigned int InputSlot::x() const
	{
		VclRequire(_resource || _source, "Slot is connected to a resource.");

		return _source ? _source->x() : _x;
	}
	unsigned int InputSlot::y() const
	{
		VclRequire(_resource || _source, "Slot is connected to a resource.");

		return _source ? _source->y() : _y;
	}
	unsigned int InputSlot::width() const
	{
		VclRequire(_resource || _source, "Slot is connected to a resource.");

		return _source ? _source->width() : _width;
	}
	unsigned int InputSlot::height() const
	{
		VclRequire(_resource || _source, "Slot is connected to a resource.");

		return _source ? _source->height() : _height;
	}

	OutputSlot::OutputSlot(const std::string& id, Task* task)
	: Slot(id, task)
	{
	}
	void OutputSlot::setResource(const std::shared_ptr<Runtime::Texture>& res)
	{
		_resource = res;
	}
	void OutputSlot::setResource(const std::shared_ptr<Runtime::Texture>& res, unsigned int width, unsigned int height)
	{
		VclRequire(implies(res, width <= (unsigned int)res->width()), "'width' is in range.");
		VclRequire(implies(res, height <= (unsigned int)res->height()), "'width' is in range.");

		_resource = res;
		_width = width;
		_height = height;
	}
	void OutputSlot::setResource(const std::shared_ptr<Runtime::Texture>& res, unsigned int x, unsigned int y, unsigned int width, unsigned int height)
	{
		VclRequire(implies(res, x + width <= (unsigned int)res->width()), "'x' + 'width' is in range.");
		VclRequire(implies(res, y + height <= (unsigned int)res->height()), "'y' + 'width' is in range.");

		_resource = res;
		_x = x;
		_y = y;
		_width = width;
		_height = height;
	}

	const Runtime::Texture* OutputSlot::resource() const
	{
		return _resource.get();
	}
	unsigned int OutputSlot::x() const
	{
		return _x;
	}
	unsigned int OutputSlot::y() const
	{
		return _y;
	}
	unsigned int OutputSlot::width() const
	{
		return _width;
	}
	unsigned int OutputSlot::height() const
	{
		return _height;
	}
}}}
