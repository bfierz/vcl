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
#include "transform.h"

 // VCL
#include <vcl/rtti/attribute.h>
#include <vcl/rtti/constructor.h>

VCL_RTTI_CTOR_TABLE_BEGIN(Transform)
	Vcl::RTTI::Constructor<System::Components::Transform, Eigen::Matrix4f>(Vcl::RTTI::Parameter<Eigen::Matrix4f>("InitialTransform"))
VCL_RTTI_CTOR_TABLE_END(Transform)

VCL_RTTI_ATTR_TABLE_BEGIN(Transform)
	Vcl::RTTI::Attribute<System::Components::Transform, const Eigen::Matrix3f&>{ "Rotation", &System::Components::Transform::rotation, &System::Components::Transform::setRotation }
VCL_RTTI_ATTR_TABLE_END(Transform)

VCL_DEFINE_METAOBJECT(System::Components::Transform)
{
	type->registerConstructors(Transform_constructor_bases);
	type->registerAttributes(Transform_attribute_bases);
}

namespace System { namespace Components
{
	Transform::Transform(const Eigen::Matrix4f& initial)
	: _transform(initial)
	{

	}

	const Eigen::Matrix3f& Transform::rotation() const
	{
		return _transform.block<3, 3>(0, 0);
	}

	void Transform::setRotation(const Eigen::Matrix3f& rotation)
	{
		_transform.block<3, 3>(0, 0) = rotation;
	}

	Eigen::Vector3f Transform::position() const
	{
		return _transform.block<3, 1>(0, 3);
	}

	void Transform::setPosition(const Eigen::Vector3f& position)
	{
		_transform.block<3, 1>(0, 3) = position;
	}
}}
