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

// GSL
#include <string_span>

// VCL
#include <vcl/core/3rdparty/any.hpp>
#include <vcl/core/convert.h>
#include <vcl/core/contract.h>

namespace Vcl { namespace RTTI
{
	template <std::size_t...> struct index_sequence {};
	template <std::size_t N, std::size_t... Is>
	struct make_index_sequence : make_index_sequence<N - 1, N - 1, Is...> {};
	template <std::size_t... Is>
	struct make_index_sequence<0u, Is...> : index_sequence<Is...> { using type = index_sequence<Is...>; };

	template<bool B> struct is_true
	{
		typedef std::false_type type;
		static const std::false_type::value_type value = std::false_type::value;
	};
	template<> struct is_true<true>
	{
		typedef std::true_type type;
		static const std::true_type::value_type value = std::true_type::value;
	};

	template < typename T, typename... Ts >
	auto head(std::tuple<T, Ts...> t)
	{
		return std::get<0>(t);
	}

	template < std::size_t... Ns, typename... Ts >
	auto tail_impl(std::index_sequence<Ns...>, std::tuple<Ts...> t)
	{
		return std::make_tuple(std::get<Ns + 1u>(t)...);
	}

	template < std::size_t... Ns, typename... Ts >
	auto tail_impl(index_sequence<Ns...>, std::tuple<Ts...> t)
	{
		return std::make_tuple(std::get<Ns + 1u>(t)...);
	}

	template < typename... Ts >
	auto tail(std::tuple<Ts...> t)
	{
		//return tail_impl(std::make_index_sequence<sizeof...(Ts)-1u>(), t);
		return tail_impl(make_index_sequence<sizeof...(Ts)-1u>(), t);
	}

	class ParameterMetaData
	{
	public:
		template<int N>
		ParameterMetaData(const char (&name)[N])
		: _name(name)
		{
		}

		ParameterMetaData(const ParameterMetaData& rhs)
		: _name(rhs._name)
		{
		}

		ParameterMetaData(ParameterMetaData&& rhs)
		{
			std::swap(_name, rhs._name);
		}

		~ParameterMetaData()
		{
		}

	public:
		gsl::cstring_span<> name() const { return _name; }

	private:
		gsl::cstring_span<> _name;
	};

	class ParameterBase
	{
	public:
		ParameterBase(ParameterMetaData meta_data, const std::type_info* info)
		: _metaData(std::move(meta_data))
		, _type(info)
		{
		}

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
		virtual linb::any pack(std::string value) const
		{
			VCL_UNREFERENCED_PARAMETER(value);
			return 0;
		}
		virtual linb::any pack(void* link) const
		{
			VCL_UNREFERENCED_PARAMETER(link);
			return nullptr;
		}
		virtual linb::any pack(std::shared_ptr<void> link) const
		{
			VCL_UNREFERENCED_PARAMETER(link);
			return std::shared_ptr<void>();
		}

	private:
		ParameterMetaData _metaData;
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
		virtual linb::any pack(std::string value) const override
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
		virtual linb::any pack(void* link) const override
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
		virtual linb::any pack(std::shared_ptr<void> link) const override
		{
			return link;
		}
	};

	//! Base class for typed constructors.
	class ConstructorBase
	{
	protected:
		ConstructorBase(int numParams)
		: _numParams(numParams)
		{
		}

	public:
		virtual ~ConstructorBase() = default;

	public:
		//! Invoke the constructor without parameters
		void* call(void* location)
		{
			return callImpl(location, {});
		}

		template<typename... Args>
		void* call(void* location, Args... args) const
		{
			return callImpl(location, { linb::any(args)... });
		}

		void* call(void* location, std::vector<linb::any> args) const
		{
			return callImpl(location, std::move(args));
		}

	public:
		//! Query the parameters of this constructor
		int numParams() const { return _numParams; }

		virtual bool hasParam(const std::string& name) const = 0;

		virtual const ParameterBase& param(int idx) const = 0;
		virtual const std::type_info* paramType(int idx) const = 0;

	protected:
		virtual void* callImpl(void* location, std::vector<linb::any>&& params) const = 0;

	private:
		//! Number of parameters for this constructor
		int _numParams;
	};

	class ConstructorSet
	{
	public:
		int size() const { return static_cast<int>(_constructors.size()); }
		const ConstructorBase& constructor(int i) const { return *_constructors[i]; }

		bool hasStandardConstructor() const { return _hasStandardConstructor;  }

	public:
		void add(std::unique_ptr<ConstructorBase> constr)
		{
			Require(implies(constr->numParams() == 0, _hasStandardConstructor == false), "Standard constructor is not set.");

			if (constr->numParams() == 0)
				_hasStandardConstructor = true;

			_constructors.push_back(std::move(constr));
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
				if (checkArgs<Args...>(constr.get()))
				{
					constr->call(location, args...);
					return;
				}
			}
		}

	private:
		template<typename... Args>
		bool checkArgs(const ConstructorBase* constr) const
		{
			if (sizeof...(Args) != constr->numParams())
				return false;

			return checkArgsImpl<Args...>(constr, make_index_sequence<sizeof...(Args)>());
		}

		template<typename... Args, size_t... S>
		bool checkArgsImpl(const ConstructorBase* constr, index_sequence<S...>) const
		{
			std::array<bool, sizeof...(Args)> results{ { checkArg<Args, S>(constr)... } };

			return std::accumulate(results.cbegin(), results.cend(), true, [] (bool a, bool b) -> bool
			{
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
		std::vector<std::unique_ptr<ConstructorBase>> _constructors;
		bool _hasStandardConstructor{ false };
	};
}}
