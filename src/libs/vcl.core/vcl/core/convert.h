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
#include <vcl/config/eigen.h>

// C++ standard library
#include <string>

// VCL
#include <vcl/core/contract.h>

namespace Vcl
{
	template<typename T>
	T convert(const std::string& value);

	template<typename T>
	std::string convert(const T& value);

	template<>
	inline std::string convert<std::string>(const std::string& value)
	{
		return value;
	}

	template<>
	inline bool convert<bool>(const std::string& value)
	{
		if (value == "true" || value == "1")
			return true;
		else if (value == "false" || value == "0")
			return false;
		
		DebugError("value not recognized");
		return false;
	}

	template<>
	inline float convert<float>(const std::string& value)
	{
		return std::stof(value);
	}

	template<>
	inline int convert<int>(const std::string& value)
	{
		return std::stoi(value);
	}
	
	template<>
	inline Eigen::Vector2f convert<Eigen::Vector2f>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		float v0 = std::stof(value, &next);      pos += next;
		float v1 = std::stof(value.substr(pos));

		return Eigen::Vector2f(v0, v1);
	}
	
	template<>
	inline Eigen::Vector3f convert<Eigen::Vector3f>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		float v0 = std::stof(value, &next);             pos += next;
		float v1 = std::stof(value.substr(pos), &next); pos += next;
		float v2 = std::stof(value.substr(pos));

		return Eigen::Vector3f(v0, v1, v2);
	}

	template<>
	inline Eigen::Vector4f convert<Eigen::Vector4f>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		float v0 = std::stof(value, &next);             pos += next;
		float v1 = std::stof(value.substr(pos), &next); pos += next;
		float v2 = std::stof(value.substr(pos), &next); pos += next;
		float v3 = std::stof(value.substr(pos));

		return Eigen::Vector4f(v0, v1, v2, v3);
	}
	
	template<>
	inline Eigen::Vector2ui convert<Eigen::Vector2ui>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		unsigned int v0 = std::stoul(value, &next);      pos += next;
		unsigned int v1 = std::stoul(value.substr(pos));

		return Eigen::Vector2ui(v0, v1);
	}
	
	template<>
	inline Eigen::Vector3ui convert<Eigen::Vector3ui>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		unsigned int v0 = std::stoul(value, &next);             pos += next;
		unsigned int v1 = std::stoul(value.substr(pos), &next); pos += next;
		unsigned int v2 = std::stoul(value.substr(pos));

		return Eigen::Vector3ui(v0, v1, v2);
	}

	template<>
	inline Eigen::Vector4ui convert<Eigen::Vector4ui>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		unsigned int v0 = std::stoul(value, &next);             pos += next;
		unsigned int v1 = std::stoul(value.substr(pos), &next);	pos += next;
		unsigned int v2 = std::stoul(value.substr(pos), &next);	pos += next;
		unsigned int v3 = std::stoul(value.substr(pos));

		return Eigen::Vector4ui(v0, v1, v2, v3);
	}
}
