/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <initializer_list>

// VCL
#include <vcl/core/preprocessor.h>
#include <vcl/core/convert.h>

#define VCL_DECLARE_ENUMS(name, n)          name = n,
#define VCL_DECLARE_ENUM_IDX_TO_VALUE(name, n) case n: return EnumType::name;
#define VCL_DECLARE_ENUM_IDX_TO_STRING(name, n) case n: return VCL_PP_STRINGIZE(name);
#define VCL_DECLARE_ENUM_TO_STRING(name, n) case EnumType::name: return VCL_PP_STRINGIZE(name);
#define VCL_DECLARE_STRING_TO_ENUM(name, n) if (value == VCL_PP_STRINGIZE(name)) { return EnumType::name; } else 

#define VCL_DECLARE_ENUM(type_name, ...)								       \
enum class type_name													       \
{																		       \
	VCL_PP_VA_EXPAND_ARGS (VCL_DECLARE_ENUMS, __VA_ARGS__)				       \
};																		       \
namespace Vcl															       \
{																		       \
	template<>																   \
	VCL_CPP_CONSTEXPR_11 unsigned int enumCount<type_name>()						   \
	{																		   \
		return VCL_PP_VA_NUM_ARGS(__VA_ARGS__);								   \
	}																		   \
	template<>																   \
	type_name enumValue<type_name>(uint32_t i)			                       \
	{                                                                          \
		using EnumType = type_name;                                            \
		switch (i)                                                             \
		{                                                                      \
		VCL_PP_VA_EXPAND_ARGS (VCL_DECLARE_ENUM_IDX_TO_VALUE, __VA_ARGS__)     \
		}                                                                      \
		return type_name(0);                                                   \
	}																		   \
	template<>																   \
	std::string enumName<type_name>(uint32_t i)		                           \
	{                                                                          \
		switch (i)                                                             \
		{                                                                      \
		VCL_PP_VA_EXPAND_ARGS (VCL_DECLARE_ENUM_IDX_TO_STRING, __VA_ARGS__)    \
		}                                                                      \
		return{};                                                              \
	}																		   \
	template<>															       \
	inline std::string to_string<type_name>(const type_name& value)            \
	{                                                                          \
		using EnumType = type_name;                                            \
		switch (value)                                                         \
		{                                                                      \
		VCL_PP_VA_EXPAND_ARGS (VCL_DECLARE_ENUM_TO_STRING, __VA_ARGS__)        \
		}                                                                      \
		return{};                                                              \
	}                                                                          \
	template<>															       \
	inline type_name from_string<type_name>(const std::string& value)          \
	{																	       \
		using EnumType = type_name;                                            \
		VCL_PP_VA_EXPAND_ARGS (VCL_DECLARE_STRING_TO_ENUM, __VA_ARGS__)        \
		{ return type_name(0); }										       \
	}																	       \
}

namespace Vcl {
	template<typename T>
	VCL_CPP_CONSTEXPR_11 unsigned int enumCount()
	{
		return 0;
	}

	template<typename T>
	T enumValue(uint32_t i)
	{
		return 0;
	}

	template<typename T>
	std::string enumName(uint32_t i)
	{
		return{};
	}
}
