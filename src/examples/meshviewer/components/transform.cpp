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
	Vcl::RTTI::Constructor<Vcl::Editor::Components::Transform, Eigen::Matrix4f>(Vcl::RTTI::Parameter<Eigen::Matrix4f>("InitialTransform"))
VCL_RTTI_CTOR_TABLE_END(Transform)

VCL_RTTI_ATTR_TABLE_BEGIN(Transform)
	Vcl::RTTI::Attribute<Vcl::Editor::Components::Transform, const Eigen::Matrix3f&>{ "Rotation", &Vcl::Editor::Components::Transform::rotation, &Vcl::Editor::Components::Transform::setRotation }
VCL_RTTI_ATTR_TABLE_END(Transform)

VCL_DEFINE_METAOBJECT(Vcl::Editor::Components::Transform)
{
	type->registerConstructors(Transform_constructor_bases);
	type->registerAttributes(Transform_attribute_bases);
}


namespace Vcl { namespace Editor { namespace Components
{
	Transform::Transform(const Eigen::Matrix4f& initial)
	: _transform(initial)
	{

	}

	const Matrix3f& Transform::rotation() const
	{
		return _transform.block<3, 3>(0, 0);
	}

	void Transform::setRotation(const Matrix3f& rotation)
	{
		_transform.block<3, 3>(0, 0) = rotation;
	}

	const Vector3f& Transform::position() const
	{
		return _transform.block<3, 1>(0, 3);
	}

	void Transform::setPosition(const Vector3f& position)
	{
		_transform.block<3, 1>(0, 3) = position;
	}
}}}
