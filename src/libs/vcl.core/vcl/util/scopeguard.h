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
#include <exception>

// VCL
#include <vcl/core/preprocessor.h>

// Implementation of uncaught_exceptions is taken from here:
// https://github.com/panaseleus/stack_unwinding/blob/master/standalone/stack_unwinding.hpp

namespace std
{
#if defined(VCL_COMPILER_MSVC) && (_MSC_VER <= 1800)
	namespace details { extern "C" char * _getptd(); }
	inline int uncaught_exceptions()
	{
		// MSVC specific. Tested on {MSVC2005SP1,MSVC2008SP1,MSVC2010SP1,MSVC2012}x{x32,x64}.
		return *(static_cast<unsigned*>(static_cast<void*>(details::_getptd() + (sizeof(void*) == 8 ? 0x100 : 0x90)))); // x32 offset - 0x90 , x64 - 0x100
	}
//#elif defined(VCL_COMPILER_GNU) || defined(VCL_COMPILER_CLANG)
#elif defined(VCL_COMPILER_CLANG)
	namespace details { extern "C" char * __cxa_get_globals(); }
	inline int uncaught_exceptions() noexcept
	{
		// Tested on {clang 3.2,GCC 3.5.6,,GCC 4.1.2,GCC 4.4.6,GCC 4.4.7}x{x32,x64}
		return *(static_cast<unsigned*>(static_cast<void*>(details::__cxa_get_globals() + (sizeof(void*) == 8 ? 0x8 : 0x4)))); // x32 offset - 0x4 , x64 - 0x8
	}
#endif
}

// Scope guard presentation:
// https://github.com/CppCon/CppCon2015/blob/master/Presentations/Declarative%20Control%20Flow/Declarative%20Control%20Flow%20-%20Andrei%20Alexandrescu%20-%20CppCon%202015.pdf

namespace Vcl { namespace Util { namespace Detail
{
	class UncaughtExceptionCounter
	{
	public:
		UncaughtExceptionCounter()
		: _exceptionCount(std::uncaught_exceptions()) {}

		UncaughtExceptionCounter(const UncaughtExceptionCounter& other)
		: _exceptionCount(other._exceptionCount) {}

		bool isNewUncaughtException() noexcept
		{
			return std::uncaught_exceptions() > _exceptionCount;
		}

	private:
		int _exceptionCount;
	};

	template <typename FunctionType>
	class ScopeGuard
	{
	public:
		explicit ScopeGuard(const FunctionType& fn)
		: _function(fn)
		{
		}

		explicit ScopeGuard(FunctionType&& fn)
		: _function(std::move(fn))
		{
		}

		ScopeGuard(ScopeGuard&& other)
		: _function(std::move(other._function))
		{
		}

		~ScopeGuard() noexcept
		{
			_function();
		}

	private:
		ScopeGuard(const ScopeGuard& other) = delete;
		void* operator new(size_t) = delete;

		FunctionType _function;
	};

	template <typename FunctionType, bool executeOnException>
	class ScopeGuardForNewException
	{
	public:
		explicit ScopeGuardForNewException(const FunctionType& fn)
		: _function(fn)
		{
		}

		explicit ScopeGuardForNewException(FunctionType&& fn)
		: _function(std::move(fn))
		{
		}

		ScopeGuardForNewException(ScopeGuardForNewException&& other)
		: _function(std::move(other._function))
		, _exceptionCounter(std::move(other._exceptionCounter))
		{
		}

		~ScopeGuardForNewException() VCL_NOEXCEPT_PARAM(executeOnException)
		{
			if (executeOnException == _exceptionCounter.isNewUncaughtException())
			{
				_function();
			}
		}

	private:
		ScopeGuardForNewException(const ScopeGuardForNewException& other) = delete;
		void* operator new(size_t) = delete;

		FunctionType _function;
		UncaughtExceptionCounter _exceptionCounter;
	};

	enum class ScopeGuardOnFail {};

	template <typename FunctionType>
	ScopeGuardForNewException<typename std::decay<FunctionType>::type, true>
		operator+(ScopeGuardOnFail, FunctionType&& fn)
	{
		return ScopeGuardForNewException<typename std::decay<FunctionType>::type, true>
		(
			std::forward<FunctionType>(fn)
		);
	}

	enum class ScopeGuardOnSuccess {};

	template <typename FunctionType>
	ScopeGuardForNewException<typename std::decay<FunctionType>::type, false>
		operator+(ScopeGuardOnSuccess, FunctionType&& fn)
	{
		return ScopeGuardForNewException<typename std::decay<FunctionType>::type, false>
		(
			std::forward<FunctionType>(fn)
		);
	}

	enum class ScopeGuardOnExit {};

	template <typename FunctionType>
	ScopeGuard<typename std::decay<FunctionType>::type>
		operator+(ScopeGuardOnExit, FunctionType&& fn)
	{
		return ScopeGuard<typename std::decay<FunctionType>::type>
		(
			std::forward<FunctionType>(fn)
		);
	}
}}}

#define VCL_SCOPE_EXIT \
  auto VCL_ANONYMOUS_VARIABLE(VCL_SCOPE_EXIT_STATE) \
  = ::Vcl::Util::Detail::ScopeGuardOnExit() + [&]() noexcept

#define VCL_SCOPE_FAIL \
  auto VCL_ANONYMOUS_VARIABLE(VCL_SCOPE_FAIL_STATE) \
  = ::Vcl::Util::Detail::ScopeGuardOnFail() + [&]() noexcept

#define VCL_SCOPE_SUCCESS \
  auto VCL_ANONYMOUS_VARIABLE(VCL_SCOPE_SUCCESS_STATE) \
  = ::Vcl::Util::Detail::ScopeGuardOnSuccess() + [&]()
