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
#include <vcl/rtti/constructor.h>
#include <vcl/rtti/metatype.h>

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
	BaseObject(const char* name) : _name(name) {}
	BaseObject(const BaseObject&) = delete;
	virtual ~BaseObject() = default;

	BaseObject& operator = (const BaseObject&) = delete;

public:
	const std::string& name() const { return _name; }
	void setName(const std::string& n) { _name = n; }

private:
	std::string _name{ "Initialized" };
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

	DerivedObject(int a, int b)
	: _size(a + b)
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

	size_t size() const { return _size; }

private:
	size_t _size;
};

TEST(RttiTest, DefaultConstructor)
{
	using namespace Vcl::RTTI;

	// Defines a default ctor
	Constructor<BaseObject> def_constr{};

	// Allocate memory for the test object
	BaseObject* obj = (BaseObject*) malloc(sizeof(BaseObject));

	// Calls the default ctor
	def_constr.call(obj);

	// Expected output
	EXPECT_EQ(std::string{ "Initialized" }, obj->name()) << "Default ctor was not called.";

	// Cleanup
	obj->~BaseObject();
	free(obj);
}

TEST(RttiTest, MultiParamConstructor)
{
	using namespace Vcl::RTTI;

	// Defines a default ctor
	Constructor<DerivedObject, int> def_constr_a
	{
		Parameter<int>("a")
	};
	Constructor<DerivedObject, int, int> def_constr_a_b
	{
		Parameter<int>("a"), Parameter<int>("b")
	};

	// Allocate memory for the test object
	auto obj_a   = (DerivedObject*)malloc(sizeof(DerivedObject));
	auto obj_a_b = (DerivedObject*)malloc(sizeof(DerivedObject));

	// Calls the default ctor
	def_constr_a.call(obj_a, 4);
	def_constr_a_b.call(obj_a_b, 4, 5);

	// Expected output
	EXPECT_EQ(4, obj_a->size()) << "ctor with one params was not called.";
	EXPECT_EQ(9, obj_a_b->size()) << "ctor with two params was not called.";

	// Cleanup
	obj_a->~DerivedObject();
	free(obj_a);
	obj_a_b->~DerivedObject();
	free(obj_a_b);
}

TEST(RttiTest, SimpleConstructor)
{
	using namespace Vcl::RTTI;

	// Defines a default ctor
	Constructor<BaseObject, const char*> def_constr
	{
		Parameter<const char*>("Name")
	};

	// Check if the parameters can be found
	EXPECT_TRUE(def_constr.hasParam("Name")) << "Parameter 'Name' is not found.";
	EXPECT_FALSE(def_constr.hasParam("NotExisting")) << "Parameter 'NotExisting' is found.";

	// Allocate memory for the test object
	BaseObject* obj = (BaseObject*)malloc(sizeof(BaseObject));

	// Calls the default ctor
	def_constr.call(obj, "String");

	// Expected output
	EXPECT_EQ(std::string{ "String" }, obj->name()) << "Default ctor was not called.";

	// Cleanup
	obj->~BaseObject();
	free(obj);
}

TEST(RttiTest, AttributeSimpleSetter)
{
	using namespace Vcl::RTTI;

	// Test object
	BaseObject obj;

	// Set an attribute
	Attribute<BaseObject, std::string> attr{ "Name", &BaseObject::name, &BaseObject::setName };
	attr.set(&obj, std::string{ "String" });

	// Expected output
	EXPECT_EQ(std::string{ "String" }, obj.name()) << "Property 'Name' was not set.";
}
