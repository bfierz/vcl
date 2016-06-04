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

// VCL
#include <vcl/core/3rdparty/any.hpp>
#include <vcl/core/convert.h>
#include <vcl/core/contract.h>
#include <vcl/rtti/constructorbase.h>

namespace Vcl { namespace RTTI
{
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

		virtual const std::type_info* paramType(int idx) const override
		{
			return getParamImpl(_parameters, idx).type();
		}

	protected:
		virtual void* callImpl(void* location, std::vector<linb::any>&& params) const override
		{
			return callImplSeq(location, std::move(params), make_index_sequence<sizeof...(Params)>());
		}

	private:
		template<size_t... S>
		void* callImplSeq(void* location, std::vector<linb::any>&& params, index_sequence<S...>) const
		{
			return call(location, getParam<Params, S>(params)...);
		}

		template<typename P, int I>
		P getParam(const std::vector<linb::any>& params) const
		{
			return extract<P>::get(params.begin()[I]);
		}

		template<typename... Ts>
		const ParameterBase& getParamImpl(const std::tuple<Ts...>& tuple, int idx) const
		{
			if (idx == 0)
				return std::get<0>(tuple);
			else
				return getParamImpl(tail(tuple), idx - 1);
		}

		template<typename P>
		const ParameterBase& getParamImpl(const std::tuple<P>& tuple, int idx) const
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

		template<typename P>
		bool hasParamImpl(const std::tuple<P>& tuple, const std::string& name) const
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

		virtual const std::type_info* paramType(int idx) const override
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
}}
