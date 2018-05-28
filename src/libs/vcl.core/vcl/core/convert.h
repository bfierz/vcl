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
	T from_string(const std::string& value)
	{
		return value;
	}

	template<typename T>
	std::string to_string(const T& value)
	{
		return value;
	}

	template<>
	inline bool from_string<bool>(const std::string& value)
	{
		if (value == "true" || value == "1")
		{
			return true;
		}
		else if (value == "false" || value == "0")
		{
			return false;
		}
		
		VclDebugError("value not recognized");
		return false;
	}
	template<>
	inline std::string to_string<bool>(const bool& value)
	{
		if (value)
		{
			return { "true" };
		}
		else
		{
			return { "false" };
		}
	}

	template<>
	inline float from_string<float>(const std::string& value)
	{
		return std::stof(value);
	}
	template<>
	inline std::string to_string<float>(const float& value)
	{
		return std::to_string(value);
	}

	template<>
	inline int from_string<int>(const std::string& value)
	{
		return std::stoi(value);
	}
	template<>
	inline std::string to_string<int>(const int& value)
	{
		return std::to_string(value);
	}
	
	template<>
	inline Eigen::Vector2f from_string<Eigen::Vector2f>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		const float v0 = std::stof(value, &next);      pos += next;
		const float v1 = std::stof(value.substr(pos));

		return Eigen::Vector2f(v0, v1);
	}
	
	template<>
	inline Eigen::Vector3f from_string<Eigen::Vector3f>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		const float v0 = std::stof(value, &next);             pos += next;
		const float v1 = std::stof(value.substr(pos), &next); pos += next;
		const float v2 = std::stof(value.substr(pos));

		return Eigen::Vector3f(v0, v1, v2);
	}

	template<>
	inline Eigen::Vector4f from_string<Eigen::Vector4f>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		const float v0 = std::stof(value, &next);             pos += next;
		const float v1 = std::stof(value.substr(pos), &next); pos += next;
		const float v2 = std::stof(value.substr(pos), &next); pos += next;
		const float v3 = std::stof(value.substr(pos));

		return Eigen::Vector4f(v0, v1, v2, v3);
	}
	
	template<>
	inline Eigen::Vector2ui from_string<Eigen::Vector2ui>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		const unsigned int v0 = std::stoul(value, &next);      pos += next;
		const unsigned int v1 = std::stoul(value.substr(pos));

		return Eigen::Vector2ui(v0, v1);
	}
	
	template<>
	inline Eigen::Vector3ui from_string<Eigen::Vector3ui>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		const unsigned int v0 = std::stoul(value, &next);             pos += next;
		const unsigned int v1 = std::stoul(value.substr(pos), &next); pos += next;
		const unsigned int v2 = std::stoul(value.substr(pos));

		return Eigen::Vector3ui(v0, v1, v2);
	}

	template<>
	inline Eigen::Vector4ui from_string<Eigen::Vector4ui>(const std::string& value)
	{
		size_t pos  = 0;
		size_t next = 0;
		const unsigned int v0 = std::stoul(value, &next);               pos += next;
		const unsigned int v1 = std::stoul(value.substr(pos), &next);	pos += next;
		const unsigned int v2 = std::stoul(value.substr(pos), &next);	pos += next;
		const unsigned int v3 = std::stoul(value.substr(pos));

		return Eigen::Vector4ui(v0, v1, v2, v3);
	}

	template<>
	inline Eigen::Matrix3f from_string<Eigen::Matrix3f>(const std::string& value)
	{
		size_t pos = 0;
		size_t next = 0;
		const float v00 = std::stof(value, &next);             pos += next;
		const float v10 = std::stof(value.substr(pos), &next); pos += next;
		const float v20 = std::stof(value.substr(pos), &next); pos += next;

		const float v01 = std::stof(value.substr(pos), &next); pos += next;
		const float v11 = std::stof(value.substr(pos), &next); pos += next;
		const float v21 = std::stof(value.substr(pos), &next); pos += next;

		const float v02 = std::stof(value.substr(pos), &next); pos += next;
		const float v12 = std::stof(value.substr(pos), &next); pos += next;
		const float v22 = std::stof(value.substr(pos));

		Eigen::Matrix3f M;
		M.col(0) << v00, v10, v20;
		M.col(1) << v01, v11, v21;
		M.col(2) << v02, v12, v22;

		return M;
	}

	template<>
	inline std::string to_string<Eigen::Matrix3f>(const Eigen::Matrix3f& value)
	{
		std::stringstream ss;

		const Eigen::IOFormat fmt{ -1, 0, ", ", "" };
		ss << value.format(fmt);

		return ss.str();
	}

	template<>
	inline Eigen::Matrix4f from_string<Eigen::Matrix4f>(const std::string& value)
	{
		size_t pos = 0;
		size_t next = 0;
		const float v00 = std::stof(value, &next);             pos += next;
		const float v10 = std::stof(value.substr(pos), &next); pos += next;
		const float v20 = std::stof(value.substr(pos), &next); pos += next;
		const float v30 = std::stof(value.substr(pos), &next); pos += next;

		const float v01 = std::stof(value.substr(pos), &next); pos += next;
		const float v11 = std::stof(value.substr(pos), &next); pos += next;
		const float v21 = std::stof(value.substr(pos), &next); pos += next;
		const float v31 = std::stof(value.substr(pos), &next); pos += next;

		const float v02 = std::stof(value.substr(pos), &next); pos += next;
		const float v12 = std::stof(value.substr(pos), &next); pos += next;
		const float v22 = std::stof(value.substr(pos), &next); pos += next;
		const float v32 = std::stof(value.substr(pos), &next); pos += next;

		const float v03 = std::stof(value.substr(pos), &next); pos += next;
		const float v13 = std::stof(value.substr(pos), &next); pos += next;
		const float v23 = std::stof(value.substr(pos), &next); pos += next;
		const float v33 = std::stof(value.substr(pos));

		Eigen::Matrix4f M;
		M.col(0) << v00, v10, v20, v30;
		M.col(1) << v01, v11, v21, v31;
		M.col(2) << v02, v12, v22, v32;
		M.col(3) << v03, v13, v23, v33;

		return M;
	}

	template<>
	inline std::string to_string<Eigen::Matrix4f>(const Eigen::Matrix4f& value)
	{
		std::stringstream ss;

		const Eigen::IOFormat fmt{ -1, 0, ", ", "" };
		ss << value.format(fmt);

		return ss.str();
	}
}
