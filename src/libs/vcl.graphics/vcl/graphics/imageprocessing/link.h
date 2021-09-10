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
#include <memory>
#include <string>
#include <vector>

// VCL
#include <vcl/graphics/runtime/resource/texture.h>

namespace Vcl { namespace Graphics { namespace ImageProcessing {
	class InputSlot;
	class OutputSlot;
	class Task;

	class Slot
	{
	public:
		Slot(const std::string& id, Task* task);
		Slot(const Slot&) = delete;
		virtual ~Slot() = default;

	public:
		const std::string& identifier() const { return _identifier; }
		Task* task() const { return _owner; }

		virtual const Runtime::Texture* resource() const = 0;

	public:
		virtual unsigned int x() const = 0;
		virtual unsigned int y() const = 0;
		virtual unsigned int width() const = 0;
		virtual unsigned int height() const = 0;

	private:
		std::string _identifier;

		// Task to which this slot belongs
		Task* _owner;
	};

	class InputSlot : public Slot
	{
	public:
		InputSlot(const std::string& id, Task* task);
		virtual ~InputSlot() = default;

	public:
		void setSource(OutputSlot* src);

		void setResource(const Runtime::Texture* input);
		void setResource(const Runtime::Texture* input, unsigned int width, unsigned int height);
		void setResource(const Runtime::Texture* input, unsigned int x, unsigned int y, unsigned int width, unsigned int height);

	public:
		virtual const Runtime::Texture* resource() const override;
		OutputSlot* source() const { return _source; }

	public:
		virtual unsigned int x() const override;
		virtual unsigned int y() const override;
		virtual unsigned int width() const override;
		virtual unsigned int height() const override;

	private:
		//! Disconnects this input slot from other task's output
		void disconnect();

	private:
		//! Link to a direct resource
		const Runtime::Texture* _resource{ nullptr };

	private:
		//! Link to a resource that is the output of another task
		OutputSlot* _source{ nullptr };

	private: // View configuration
		unsigned int _x{ 0 };
		unsigned int _y{ 0 };
		unsigned int _width{ 0 };
		unsigned int _height{ 0 };
	};

	class OutputSlot : public Slot
	{
		friend class InputSlot;

	public:
		OutputSlot(const std::string& id, Task* task);
		virtual ~OutputSlot() = default;

	public:
		const std::vector<InputSlot*> connections() const { return _connections; }

	public:
		void setResource(const std::shared_ptr<Runtime::Texture>& res);
		void setResource(const std::shared_ptr<Runtime::Texture>& res, unsigned int width, unsigned int height);
		void setResource(const std::shared_ptr<Runtime::Texture>& res, unsigned int x, unsigned int y, unsigned int width, unsigned int height);

	public:
		virtual const Runtime::Texture* resource() const override;

	public:
		virtual unsigned int x() const override;
		virtual unsigned int y() const override;
		virtual unsigned int width() const override;
		virtual unsigned int height() const override;

	private:
		//! List of input connected to this output slot
		std::vector<InputSlot*> _connections;

		// Resource used as output
		std::shared_ptr<Runtime::Texture> _resource;

	private: // View configuration
		unsigned int _x{ 0 };
		unsigned int _y{ 0 };
		unsigned int _width{ 0 };
		unsigned int _height{ 0 };
	};
}}}
