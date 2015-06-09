/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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
#include <atomic>
#include <cstdint>
#include <memory>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Core
{
	class ref_cnt
	{
	public:
		ref_cnt()
		{
			_references = false;
		}

		void setValid()
		{
			_references = true;
		}
		void setInvalid()
		{
			_references = false;
		}

		bool valid() const
		{
			return _references;
		}

	private:
		std::atomic_bool _references;
	};

	template<typename T>
	class owner_ptr
	{
		template<typename T> friend class ref_ptr;
	public:
		owner_ptr() = default;
		owner_ptr(T* ptr)
		: _ptr(ptr)
		{
			_cnt = std::make_shared<ref_cnt>();
			_cnt->setValid();
		}
		owner_ptr(const owner_ptr&) = delete;
		owner_ptr(owner_ptr&& rhs)
		{
			std::swap(_ptr, rhs._ptr);
			std::swap(_cnt, rhs._cnt);
		}

		~owner_ptr()
		{
			_cnt->setInvalid();
			_ptr = nullptr;
		}

		void reset(T* ptr = nullptr)
		{
			_cnt = std::make_shared<ref_cnt>();
			_cnt->setValid();
			_ptr = ptr;
		}

		operator bool() const
		{
			return _ptr;
		}

		T& operator*()
		{
			return *_ptr;
		}

		T* operator->() const
		{
			return _ptr;
		}

		T* get() const { return _ptr; }

	public: // Access
		long use_count() const
		{
			return _cnt.use_count() - 1;
		}

	private:
		T* _ptr{ nullptr };
		std::shared_ptr<ref_cnt> _cnt;
	};

	template<typename T>
	class ref_ptr
	{
	public:
		ref_ptr() = default;

		ref_ptr(const owner_ptr<T>& ptr)
		: _ptr(ptr.get())
		, _cnt(ptr._cnt)
		{
		}

		~ref_ptr()
		{
			_ptr = nullptr;
		}

		void reset()
		{
			_cnt.reset();
			_ptr = nullptr;
		}

		void reset(const owner_ptr<T>& ptr)
		{
			_cnt = ptr._cnt;
			_ptr = ptr.get();
		}

		operator bool() const
		{
			return _ptr && _cnt;
		}

		T& operator*()
		{
			return *_ptr;
		}

		T* operator->() const
		{
			return _ptr;
		}

		T* get() const { return _ptr; }

	private:
		T* _ptr{ nullptr };
		std::shared_ptr<ref_cnt> _cnt;
	};

	template<typename T, typename... Args>
	owner_ptr<T> make_owner(Args&&... args)
	{
		return owner_ptr<T>(new T(std::forward<Args>(args)...));
	}
}}
