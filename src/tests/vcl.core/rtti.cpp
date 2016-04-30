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

// VCL configuration
#include <vcl/config/global.h>

// Include the relevant parts from the library
#include <vcl/rtti/attribute.h>

// C++ standard library
#include <random>
#include <vector>

// Google test
#include <gtest/gtest.h>

// Test classes
class BaseObject
{
	//VCL_DECLARE_METAOBJECT(BaseObject)

public:
	BaseObject() = default;
	BaseObject(const BaseObject&) = delete;
	virtual ~BaseObject() = default;

	BaseObject& operator = (const BaseObject&) = delete;

public:
	const std::string& name() const { return _name; }
	void setName(const std::string& n) { _name = n; }

private:
	std::string _name;
};

class AdditionalBase
{
	//VCL_DECLARE_METAOBJECT(AdditionalBase)

private:
	std::string _additionalName;
};

class DerivedObject : public BaseObject, public AdditionalBase
{
	//VCL_DECLARE_METAOBJECT(DerivedObject)

public:
	DerivedObject()
		: _size(42)
	{
	}

	DerivedObject(int a)
		: _size(a)
	{
	}

	const float& getFloat() const
	{
		return (const float&) _size;
	}

	void setFloat(const float& f)
	{
		_size = (size_t) f;
	}

private:
	size_t _size;
};


TEST(RttiTest, Attribute)
{
	using namespace Vcl::RTTI;

	BaseObject obj;

	Attribute<BaseObject, std::string> attr{ "Name", &BaseObject::name, &BaseObject::setName };
	attr.set(&obj, std::string{ "String" });
}
