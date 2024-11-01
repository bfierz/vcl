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
#include <array>
#include <initializer_list>
#include <unordered_map>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>
#include <utility>

// Abseil
#include <absl/utility/utility.h>

// VCL
#include <vcl/core/container/array.h>
#include <vcl/core/any.h>
#include <vcl/core/convert.h>
#include <vcl/core/contract.h>
#include <vcl/core/span.h>
#include <vcl/core/string_view.h>

namespace Vcl { namespace RTTI {
	template<typename T, typename... Ts>
	auto& head(std::tuple<T, Ts...> t)
	{
		return std::get<0>(t);
	}

	template<std::size_t... Ns, typename... Ts>
	auto tail_impl(absl::index_sequence<Ns...>, const std::tuple<Ts...>& t)
	{
		return std::forward_as_tuple(std::get<Ns + 1u>(t)...);
	}

	template<typename... Ts>
	auto tail(const std::tuple<Ts...>& t)
	{
		return tail_impl(absl::make_index_sequence<sizeof...(Ts) - 1u>(), t);
	}

	class ParameterMetaData
	{
	public:
		template<int N>
		constexpr ParameterMetaData(const char (&name)[N])
		: _name(name, N - 1)
		{
		}

		constexpr ParameterMetaData(const ParameterMetaData& rhs) = default;
		constexpr ParameterMetaData(ParameterMetaData&& rhs) = default;

	public:
		stdext::string_view name() const { return _name; }

	private:
		const stdext::string_view _name;
	};

	class ParameterBase
	{
	public:
		ParameterBase(ParameterMetaData meta_data, const std::type_info* info)
		: _metaData(std::move(meta_data))
		, _type(info)
		{
		}
		ParameterBase(const ParameterBase&) = default;
		ParameterBase(ParameterBase&&) = default;
		virtual ~ParameterBase() = default;

	public:
		const ParameterMetaData& data() const
		{
			return _metaData;
		}

		const std::type_info* type() const
		{
			return _type;
		}

	public:
		virtual stdext::any pack(std::string value) const
		{
			VCL_UNREFERENCED_PARAMETER(value);
			return 0;
		}
		virtual stdext::any pack(void* link) const
		{
			VCL_UNREFERENCED_PARAMETER(link);
			return nullptr;
		}
		virtual stdext::any pack(std::shared_ptr<void> link) const
		{
			VCL_UNREFERENCED_PARAMETER(link);
			return std::shared_ptr<void>();
		}

	private:
		//! Meta data describing the parameter
		const ParameterMetaData _metaData;

		//! C++ RTTI object
		const std::type_info* _type;
	};

	template<typename T>
	class Parameter : public ParameterBase
	{
	public:
		Parameter(ParameterMetaData meta_data)
		: ParameterBase(std::move(meta_data), &typeid(T))
		{
		}

	public:
		virtual stdext::any pack(std::string value) const override
		{
			return from_string<T>(value);
		}
	};

	template<typename T>
	class Parameter<T*> : public ParameterBase
	{
	public:
		Parameter(ParameterMetaData meta_data)
		: ParameterBase(std::move(meta_data), &typeid(T*))
		{
		}

	public:
		virtual stdext::any pack(void* link) const override
		{
			return link;
		}
	};

	template<typename T>
	class Parameter<std::shared_ptr<T>> : public ParameterBase
	{
	public:
		Parameter(ParameterMetaData meta_data)
		: ParameterBase(std::move(meta_data), &typeid(std::shared_ptr<T>))
		{
		}

	public:
		virtual stdext::any pack(std::shared_ptr<void> link) const override
		{
			return link;
		}
	};

	//! Base class for typed constructors.
	class ConstructorBase
	{
	protected:
		constexpr ConstructorBase(int numParams)
		: _numParams(numParams)
		{
		}

	public:
		virtual ~ConstructorBase() = default;

		//! Invoke the constructor without parameters
		void* call(void* location) const
		{
			return callImpl(location, {});
		}

		template<typename... Args>
		void* call(void* location, Args... args) const
		{
			auto any_args = std::make_array<stdext::any>(args...);
			return callImpl(location, stdext::make_span(any_args));
		}

		void* call(void* location, stdext::span<stdext::any> args) const
		{
			return callImpl(location, args);
		}

	public:
		//! Query the parameters of this constructor
		int numParams() const { return _numParams; }

		template<size_t N>
		bool hasParam(const char (&name)[N])
		{
			return hasParam(name, N - 1);
		}

		virtual bool hasParam(const stdext::string_view name) const = 0;

		virtual const ParameterBase& param(int idx) const = 0;
		virtual const std::type_info* paramType(int idx) const = 0;

	protected:
		virtual void* callImpl(void* location, stdext::span<stdext::any> params) const = 0;

	private:
		//! Number of parameters for this constructor
		int _numParams;
	};

	class ConstructorSet
	{
	public:
		int size() const { return static_cast<int>(_constructors.size()); }
		const ConstructorBase& constructor(int i) const { return *_constructors[i]; }

		bool hasStandardConstructor() const { return _hasStandardConstructor; }

	public:
		template<size_t N>
		VCL_STRONG_INLINE void set(std::array<const ConstructorBase*, N>& constructors)
		{
			_constructors = stdext::make_span(constructors);
			for (auto c : constructors)
				if (c->numParams() == 0)
					_hasStandardConstructor = true;
		}
		VCL_STRONG_INLINE void set(stdext::span<std::unique_ptr<ConstructorBase>> constructors)
		{
			_constructors = { reinterpret_cast<const ConstructorBase**>(constructors.data()), constructors.size() };
			for (const auto& c : constructors)
				if (c->numParams() == 0)
					_hasStandardConstructor = true;
		}

		void call(void* location) const
		{
			for (auto& constr : _constructors)
			{
				if (constr->numParams() == 0)
				{
					constr->call(location);
					return;
				}
			}

			throw std::runtime_error{ "Default ctor was not found." };
		}

		template<typename... Args>
		void call(void* location, Args... args) const
		{
			for (auto& constr : _constructors)
			{
				if (checkArgs<Args...>(constr))
				{
					constr->call(location, args...);
					return;
				}
			}

			throw std::runtime_error{ "No compatible ctor was not found." };
		}

	private:
		template<typename... Args>
		bool checkArgs(const ConstructorBase* constr) const
		{
			if (sizeof...(Args) != constr->numParams())
				return false;

			return checkArgsImpl<Args...>(constr, absl::make_index_sequence<sizeof...(Args)>());
		}

		template<typename... Args, size_t... S>
		bool checkArgsImpl(const ConstructorBase* constr, absl::index_sequence<S...>) const
		{
			std::array<bool, sizeof...(Args)> results{ { checkArg<Args, S>(constr)... } };

			return std::accumulate(results.cbegin(), results.cend(), true, [](bool a, bool b) -> bool {
				return a && b;
			});
		}

		template<typename T, int I>
		bool checkArg(const ConstructorBase* constr) const
		{
			if (I < constr->numParams() && (constr->paramType(I)->before(typeid(T)) || *constr->paramType(I) == typeid(T)))
				return true;
			else
				return false;
		}

	private:
		//! List of registered ctor's
		stdext::span<const ConstructorBase*> _constructors;

		//! Indicate whether a standard ctor is available
		bool _hasStandardConstructor{ false };
	};
}}
