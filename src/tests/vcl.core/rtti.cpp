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
#include <vcl/rtti/factory.h>
#include <vcl/rtti/metatypeconstructor.h>
#include <vcl/rtti/metatype.h>
#include <vcl/rtti/serializer.h>

// C++ standard library
#include <random>
#include <vector>

// Google test
#include <gtest/gtest.h>

// Moc classes
class Serializer : public Vcl::RTTI::Serializer
{
public:
	virtual void beginType(const char* name, int version)
	{

	}

	virtual void endType()
	{

	}

	virtual void writeAttribute(const char* name, const std::string& value) override
	{
		_attributes[name] = value;
	}

public:
	const std::string& attribute(const std::string& name) const
	{
		if (_attributes.find(name) != _attributes.end())
		{
			return _attributes.find(name)->second;
		}

		return "";
	}

private:
	std::unordered_map<std::string, std::string> _attributes;
};

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

	BaseObject(Vcl::RTTI::Deserializer* d)
	{
		auto type = Vcl::RTTI::MetaTypeSingleton<BaseObject>::get();

		for (const auto& attr : type->attributes())
		{
			if (d->hasAttribute(attr->name()))
			{
				attr->set(this, d->readAttribute(attr->name()));
			}
		}
	}

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
	std::string _additionalName{ "NoValue" };
};

class DerivedObject : public BaseObject, public AdditionalBase
{
	VCL_DECLARE_METAOBJECT(DerivedObject)

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

	BaseObject* ownedObj() const { return _ownedObj.get(); }
	void setOwnedObj(std::unique_ptr<BaseObject> obj) { _ownedObj = std::move(obj); }

	size_t size() const { return _size; }

private:
	size_t _size;

	std::unique_ptr<BaseObject> _ownedObj;
};

VCL_DEFINE_METAOBJECT(BaseObject)
{
	type->addConstructor();
	type->addConstructor(Parameter<const char*>("Name"));
	type->addAttribute("Name", &BaseObject::name, &BaseObject::setName);
}

VCL_DEFINE_METAOBJECT(AdditionalBase)
{

}

VCL_DEFINE_METAOBJECT(DerivedObject)
{
	type->addConstructor();
	type->addAttribute("OwnedMember", &DerivedObject::ownedObj, &DerivedObject::setOwnedObj);
}

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

TEST(RttiTest, SimpleConstructableType)
{
	using namespace Vcl::RTTI;

	// Build the constructable type
	ConstructableType<BaseObject> type{ "BaseObject", sizeof(BaseObject), alignof(BaseObject) };
	type.addConstructor();
	type.addConstructor(Parameter<const char*>("Name"));
	type.addAttribute("Name", &BaseObject::name, &BaseObject::setName);

	// Build a test object
	void* obj_mem = type.allocate();
	type.Type::construct(obj_mem);

	auto obj = (BaseObject*) obj_mem;

	// Check the expected output
	EXPECT_TRUE(type.hasAttribute("Name")) << "Attribute 'Name' is not found.";
	EXPECT_EQ(std::string{ "Initialized" }, obj->name()) << "Default ctor was not called.";

	type.destruct(obj_mem);
	type.deallocate(obj_mem);
}

TEST(RttiTest, DerivedConstructableType)
{
	using namespace Vcl::RTTI;

	// Build the constructable type
	ConstructableType<BaseObject> type{ "BaseObject", sizeof(BaseObject), alignof(BaseObject) };
	type.addConstructor();
	type.addConstructor(Parameter<const char*>("Name"));
	type.addAttribute("Name", &BaseObject::name, &BaseObject::setName);

	ConstructableType<DerivedObject> type_d{ "DerivedObject", sizeof(DerivedObject), alignof(DerivedObject) };
	type_d.inherit<BaseObject>();
	type_d.inherit<AdditionalBase>();
	type_d.addConstructor();
	type_d.addConstructor(Parameter<int>("a"));
	type_d.addConstructor(Parameter<int>("a"), Parameter<int>("b"));

	// Build a test object
	void* obj_mem = type_d.allocate();
	type_d.Type::construct(obj_mem, 42, 4);

	auto obj = (DerivedObject*)obj_mem;

	// Check the expected output
	EXPECT_EQ(46, obj->size()) << "Constructor was not called correctly.";
	EXPECT_TRUE(type_d.isA(&type)) << "Inheritance is not constructed correctly.";

	type_d.destruct(obj_mem);
	type_d.deallocate(obj_mem);
}

TEST(RttiTest, SimpleFactoryUse)
{
	using namespace Vcl::RTTI;

	// Create a new type
	auto obj = (BaseObject*) Factory::create("BaseObject", "Param0");

	EXPECT_EQ(std::string{ "Param0" }, obj->name()) << "Default ctor was not called.";

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

TEST(RttiTest, AttributeOwnedMember)
{
	using namespace Vcl::RTTI;

	// Test object
	DerivedObject obj;
	auto mem = std::make_unique<BaseObject>("String");

	// Set an attribute
	obj.metaType()->attribute("OwnedMember")->set(&obj, mem.release());

	// Expected output
	EXPECT_EQ(std::string{ "String" }, obj.ownedObj()->name()) << "Property 'Name' was not set.";
}

TEST(RttiTest, SimpleObjectSerialization)
{
	using namespace Vcl::RTTI;

	// Test object
	BaseObject obj;

	// Serialize
	::Serializer ser;

	auto type = vcl_meta_type(obj);
	type->serialize(ser, &obj);

	// Check
	EXPECT_EQ(std::string{ "Initialized" }, ser.attribute("Name")) << "Attribute was serialized.";
}
