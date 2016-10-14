/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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

// C++ standard library
#include <iostream>

// VCL
#include <vcl/rtti/attribute.h>
#include <vcl/rtti/metatype.h>
#include <vcl/rtti/metatypelookup.h>
#include <vcl/rtti/metatypeconstructor.inl>

// Test classes
#include "baseobject.h"

class AdditionalBase
{
	VCL_DECLARE_ROOT_METAOBJECT(AdditionalBase)

private:
	std::string _additionalName{ "NoValue" };
};

class DerivedObject : public BaseObject
{
	VCL_DECLARE_METAOBJECT(DerivedObject)

public:
	DerivedObject()
		: _size(42)
	{
	}

	DerivedObject(int a)
		: _size((float)a)
	{
	}

	DerivedObject(int a, int b)
		: _size((float)(a + b))
	{
	}

	const float& getFloat() const
	{
		return (const float&)_size;
	}

	void setFloat(const float& f)
	{
		_size = f;
	}

	BaseObject* ownedObj() const { return _ownedObj.get(); }
	void setOwnedObj(std::unique_ptr<BaseObject> obj) { _ownedObj = std::move(obj); }

	size_t size() const { return (size_t)_size; }

private:
	float _size;

	std::unique_ptr<BaseObject> _ownedObj;
};

class ComplexDerivedObject : public DerivedObject, public AdditionalBase
{
	VCL_DECLARE_METAOBJECT(ComplexDerivedObject)
};

VCL_RTTI_CTOR_TABLE_BEGIN(AdditionalBase)
	Vcl::RTTI::Constructor<AdditionalBase>()
VCL_RTTI_CTOR_TABLE_END(AdditionalBase)

VCL_DEFINE_METAOBJECT(AdditionalBase)
{
	type->registerConstructors(AdditionalBase_constructor_bases);
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

VCL_RTTI_BASES(ComplexDerivedObject, DerivedObject, AdditionalBase)

VCL_RTTI_CTOR_TABLE_BEGIN(ComplexDerivedObject)
	Vcl::RTTI::Constructor<ComplexDerivedObject>()
VCL_RTTI_CTOR_TABLE_END(ComplexDerivedObject)

VCL_DEFINE_METAOBJECT(ComplexDerivedObject)
{
	type->registerBaseClasses(ComplexDerivedObject_parents);
	type->registerConstructors(ComplexDerivedObject_constructor_bases);
}

int main(int, char**)
{
	using namespace Vcl::RTTI;

	// Test object
	ComplexDerivedObject obj;
	auto mem = std::make_unique<BaseObject>("String");

	// Set an attribute
	obj.metaType()->attribute("OwnedMember")->set(&obj, mem.release());

	// Print the attribute
	std::cout << "Attribute assign to property 'OwnedMember': " << obj.ownedObj()->name() << "\n";

	return 0;
}
