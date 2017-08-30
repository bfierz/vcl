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
#include <vcl/rtti/metatype.h>
#include <vcl/rtti/metatypeconstructor.inl>
#include <vcl/rtti/serializer.h>

// C++ standard library
#include <random>
#include <vector>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>

// JSON library
#include <json.hpp>
VCL_END_EXTERNAL_HEADERS

// For convenience
using json = nlohmann::json;

// Moc classes
class Serializer : public Vcl::RTTI::Serializer
{
public:
	virtual void beginType(const gsl::cstring_span<> name, int version) override
	{
		_objects.emplace_back(_attrib, json{ {"Type", name.data() },  { "Version", version } });
	}

	virtual void endType() override
	{
		auto attrib_obj = std::move(_objects.back());
		_objects.pop_back();

		if (!_objects.empty())
		{
			_objects.back().second["Attributes"][attrib_obj.first] = std::move(attrib_obj.second);
		}
		else
		{
			_storage = std::move(attrib_obj.second);
		}
	}

	virtual void writeAttribute(const gsl::cstring_span<> name, const gsl::cstring_span<> value) override
	{
		_attrib = name.data();
		if (!_objects.empty())
		{
			_objects.back().second["Attributes"][name.data()] = value.data();
		}

		_attributes[name.data()] = value.data();
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

	void print()
	{
		std::cout << _storage.dump() << std::endl;
	}

private:
	std::unordered_map<std::string, std::string> _attributes;

	/// Object storage
	json _storage;

	/// Stack of currently edited objects
	std::vector<std::pair<const std::string, json>> _objects;

	/// Current attribute
	std::string _attrib;
};

class Deserializer : public Vcl::RTTI::Deserializer
{
public:
	Deserializer(const json& storage)
	: _storage(storage)
	{
	}

public:
	virtual void beginType(const gsl::cstring_span<> name) override
	{
		if (name.empty())
			_object_stack.emplace_back(&_storage);
		else
			_object_stack.emplace_back(&(*_object_stack.back())["Attributes"][name.data()]);
	}

	virtual void endType() override
	{
		_object_stack.pop_back();

	}

	virtual std::string readType() override
	{
		return (*_object_stack.back())["Type"];
	}

	virtual bool hasAttribute(const gsl::cstring_span<> name) override
	{
		auto& attribs = (*_object_stack.back())["Attributes"];
		return attribs.find(name.data()) != attribs.cend();
	}

	virtual std::string readAttribute(const gsl::cstring_span<> name) override
	{
		return (*_object_stack.back())["Attributes"][name.data()];
	}

private:
	const json& _storage;

	std::vector<const json*> _object_stack;

};

// Test classes
class BaseObject
{
	VCL_DECLARE_ROOT_METAOBJECT(BaseObject)

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
	VCL_DECLARE_ROOT_METAOBJECT(AdditionalBase)

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
	: _size((float) a)
	{
	}

	DerivedObject(int a, int b)
	: _size((float) (a + b))
	{
	}

	const float& getFloat() const
	{
		return (const float&) _size;
	}

	void setFloat(const float& f)
	{
		_size = f;
	}

	BaseObject* ownedObj() const { return _ownedObj.get(); }
	void setOwnedObj(std::unique_ptr<BaseObject> obj) { _ownedObj = std::move(obj); }

	size_t size() const { return (size_t) _size; }

private:
	float _size;

	std::unique_ptr<BaseObject> _ownedObj;
};


VCL_RTTI_CTOR_TABLE_BEGIN(BaseObject)
	Vcl::RTTI::Constructor<BaseObject>(),
	Vcl::RTTI::Constructor<BaseObject, const char*>(Vcl::RTTI::Parameter<const char*>("Name"))
VCL_RTTI_CTOR_TABLE_END(BaseObject)

VCL_RTTI_ATTR_TABLE_BEGIN(BaseObject)
	Vcl::RTTI::Attribute<BaseObject, const std::string&>{ "Name", &BaseObject::name, &BaseObject::setName }
VCL_RTTI_ATTR_TABLE_END(BaseObject)

VCL_DEFINE_METAOBJECT(BaseObject)
{
	type->registerConstructors(BaseObject_constructor_bases);
	type->registerAttributes(BaseObject_attribute_bases);
}

VCL_DEFINE_METAOBJECT(AdditionalBase)
{
}

VCL_RTTI_BASES(DerivedObject, BaseObject)

VCL_RTTI_CTOR_TABLE_BEGIN(DerivedObject)
Vcl::RTTI::Constructor<DerivedObject>()
VCL_RTTI_CTOR_TABLE_END(DerivedObject)

VCL_RTTI_ATTR_TABLE_BEGIN(DerivedObject)
Vcl::RTTI::Attribute<DerivedObject, std::unique_ptr<BaseObject>>{ "OwnedMember", &DerivedObject::ownedObj, &DerivedObject::setOwnedObj }
VCL_RTTI_ATTR_TABLE_END(DerivedObject)

VCL_DEFINE_METAOBJECT(DerivedObject)
{
	type->registerBaseClasses(DerivedObject_parents);
	type->registerConstructors(DerivedObject_constructor_bases);
	type->registerAttributes(DerivedObject_attribute_bases);
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
	EXPECT_EQ(4u, obj_a->size()) << "ctor with one params was not called.";
	EXPECT_EQ(9u, obj_a_b->size()) << "ctor with two params was not called.";

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
	DynamicConstructableType<BaseObject> type{ "BaseObject", sizeof(BaseObject), alignof(BaseObject) };
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
	DynamicConstructableType<BaseObject> type{ "BaseObject", sizeof(BaseObject), alignof(BaseObject) };
	type.addConstructor();
	type.addConstructor(Parameter<const char*>("Name"));
	type.addAttribute("Name", &BaseObject::name, &BaseObject::setName);

	DynamicConstructableType<DerivedObject> type_d{ "DerivedObject", sizeof(DerivedObject), alignof(DerivedObject) };
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
	EXPECT_EQ(46u, obj->size()) << "Constructor was not called correctly.";
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
	Attribute<BaseObject, const std::string&> attr{ "Name", &BaseObject::name, &BaseObject::setName };
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

TEST(RttiTest, SimpleObjectDeserialization)
{
	using namespace Vcl::RTTI;
	
	// Serialize
	json storage = 
	{
		{"Type", "BaseObject"},
		{"Version", 1},
		{
			"Attributes", 
			{
				{"Name", "Loaded"}
			}
		}
	};
	::Deserializer loader(storage);
	loader.beginType("");

	void* obj_store = Factory::create(loader.readType());
	BaseObject& obj = *reinterpret_cast<BaseObject*>(obj_store);

	auto type = vcl_meta_type(obj);
	type->deserialize(loader, obj_store);
	loader.endType();

	// Check
	EXPECT_EQ(std::string{ "Loaded" }, obj.name()) << "Attribute was deserialized.";
}

TEST(RttiTest, ComplexObjectSerialization)
{
	using namespace Vcl::RTTI;

	// Test object
	DerivedObject obj;

	// Serialize
	::Serializer ser;

	auto type = vcl_meta_type(obj);
	type->serialize(ser, &obj);

	// Check
	EXPECT_EQ(std::string{ "Initialized" }, ser.attribute("Name")) << "Attribute was serialized.";
}

TEST(RttiTest, ComplexObjectDeserialization)
{
	using namespace Vcl::RTTI;

	// Serialize
	json storage =
	{
		{ "Type", "DerivedObject" },
		{ "Version", 1 },
		{
			"Attributes",
			{
				{ "Name", "OuterLoaded" },
				{
					"OwnedMember", 
					{
						{ "Type", "BaseObject" },
						{ "Version", 1 },
						{
							"Attributes",
							{
								{ "Name", "Loaded" }
							}
						}
					}
				}
			}
		}
	};
	::Deserializer loader(storage);

	loader.beginType("");
	void* obj_store = Factory::create(loader.readType());
	DerivedObject& obj = *reinterpret_cast<DerivedObject*>(obj_store);

	auto type = vcl_meta_type(obj);
	type->deserialize(loader, obj_store);
	loader.endType();

	// Check
	EXPECT_EQ(std::string{ "OuterLoaded" }, obj.name()) << "Attribute was deserialized.";
	EXPECT_EQ(std::string{ "Loaded" }, obj.ownedObj()->name()) << "Attribute was deserialized.";
}
