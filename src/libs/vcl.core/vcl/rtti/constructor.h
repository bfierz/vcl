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
#include <vector>
#include <utility>

// VCL
#include <string_span.h>
#include <vcl/core/3rdparty/any.hpp>
#include <vcl/core/contract.h>

namespace Vcl { namespace RTTI
{
	template<int ...> struct index_sequence {};
	template<int N, int ...S> struct make_index_sequence : make_index_sequence<N - 1, N - 1, S...> {};
	template<int ...S> struct make_index_sequence<0, S...> { typedef index_sequence<S...> type; };

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

	template < typename... Ts >
	auto tail(std::tuple<Ts...> t)
	{
		return tail_impl(std::make_index_sequence<sizeof...(Ts)-1u>(), t);
	}

	template<typename T>
	struct extract
	{
		static T get(const linb::any& value)
		{
			return linb::any_cast<T>(value);
		}
	};
	template<typename T>
	struct extract<std::shared_ptr<T>>
	{
		static std::shared_ptr<T> get(const linb::any& value)
		{
			return std::static_pointer_cast<T>(std::move(linb::any_cast<std::shared_ptr<void>>(value)));
		}
	};

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
		ParameterBase(ParameterMetaData meta_data, const type_info* info)
		: _metaData(std::move(meta_data))
		, _type(info)
		{
		}

	public:
		const ParameterMetaData& data() const
		{
			return _metaData;
		}

		const type_info* type() const
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
		const type_info* _type;
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
			return convert<T>(value);
		}
	};

	template<typename T>
	class Parameter<T*> : public ParameterBase
	{
	public:
		Parameter(ParameterMetaData meta_data)
			: ParameterBase(std::move(meta_data), &typeid(T))
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
		: ParameterBase(std::move(meta_data), &typeid(T))
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
		virtual const type_info* paramType(int idx) const = 0;

	protected:
		virtual void* callImpl(void* location, std::vector<linb::any>&& params) const = 0;

	private:
		//! Number of parameters for this constructor
		int _numParams;
	};

	template<typename T, typename... Params>
	class Constructor : public ConstructorBase
	{
		static const int NumParams = sizeof...(Params);

	public:
		Constructor(Parameter<Params>... desc)
		: ConstructorBase(NumParams)
		, _parameters(std::make_tuple(std::forward<Parameter<Params>>(desc)...))
		{
		}

		virtual ~Constructor() = default;

	public:
		T* call(void* location, Params... params) const
		{
			return new(location) T(std::forward<Params>(params)...);
		}

		virtual bool hasParam(const std::string& name) const override
		{
			return hasParamImpl(_parameters, name);
		}

		virtual const ParameterBase& param(int idx) const override
		{
			return getParamImpl(_parameters, idx);
		}

		virtual const type_info* paramType(int idx) const override
		{
			return getParamImpl(_parameters, idx).type();
		}

	protected:
		virtual void* callImpl(void* location, std::vector<linb::any>&& params) const override
		{
			return callImplSeq(location, std::move(params), typename make_index_sequence<sizeof...(Params)>::type());
		}

	private:
		template<int... S>
		void* callImplSeq(void* location, std::vector<linb::any>&& params, index_sequence<S...>) const
		{
			return call(location, getParam<Params, S>(params)...);
		}

		template<typename T, int I>
		T getParam(const std::vector<linb::any>& params) const
		{
			return extract<T>::get(params.begin()[I]);
		}

		template<typename... Ts>
		const ParameterBase& getParamImpl(const std::tuple<Ts...>& tuple, int idx) const
		{
			if (idx == 0)
				return std::get<0>(tuple);
			else
				return getParamImpl(tail(tuple), idx - 1);
		}

		template<typename T>
		const ParameterBase& getParamImpl(const std::tuple<T>& tuple, int idx) const
		{
			Require(idx == 0, "Tuple with one element has index 0.");

			return std::get<0>(tuple);
		}

		template<typename... Ts>
		bool hasParamImpl(const std::tuple<Ts...>& tuple, const std::string& name) const
		{
			if (head(_parameters).data().name() == name)
				return true;
			else
				return hasParamImpl(tail(tuple), name);
		}

		template<typename T>
		bool hasParamImpl(const std::tuple<T>& tuple, const std::string& name) const
		{
			return std::get<0>(tuple).data().name() == name;
		}

	private:
		std::tuple<Parameter<Params>...> _parameters;
	};

	template<typename T>
	class Constructor<T> : public ConstructorBase
	{
		static const int NumParams = 0;

	public:
		Constructor()
		: ConstructorBase(NumParams)
		{
		}

		virtual ~Constructor() = default;

	public:
		T* call(void* location)
		{
			return new(location) T;
		}

		virtual bool hasParam(const std::string& name) const override
		{
			VCL_UNREFERENCED_PARAMETER(name);
			return false;
		}

		virtual const ParameterBase& param(int idx) const override
		{
			VCL_UNREFERENCED_PARAMETER(idx);
			return _default;
		}

		virtual const type_info* paramType(int idx) const override
		{
			VCL_UNREFERENCED_PARAMETER(idx);
			return nullptr;
		}

	protected:
		virtual void* callImpl(void* location, std::vector<linb::any>&& params) const override
		{
			Require(params.size() == 0, "No parameters supplied.");

			return new(location) T;
		}

		ParameterBase _default{ { "Default" }, 0 };
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

			return checkArgsImpl<Args...>(constr, typename make_index_sequence<sizeof...(Args)>::type());
		}

		template<typename... Args, int... S>
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
