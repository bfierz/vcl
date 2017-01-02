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
#include "entityadapter.h"

#include "components/transform.h"

#include "componentadapter.h"

namespace Editor
{
	EntityAdapter::EntityAdapter(const QString& name, const Vcl::Components::Entity& entity)
	: _name(name)
	, _entity(entity)
	{
	}

	QString EntityAdapter::name() const
	{
		return _name;
	}

	void EntityAdapter::setName(const QString& name)
	{
		if (name != _name)
		{
			_name = name;
		}
	}

	QList<QObject*> EntityAdapter::components() const
	{
		if (_components.empty())
		{
			const auto& component_stores = _entity.manager()->uniqueComponents();
			for (const auto& id_store : component_stores)
			{
				const auto* type = id_store.second->type();
				if (type && type->isA(vcl_meta_type<System::Components::Transform>()))
				{
					_components << new TransformComponentAdapter();
				}
			}
		}

		return _components;
	}

	EntityAdapterModel::EntityAdapterModel(QObject* parent)
	: QAbstractListModel(parent)
	{
	}

	void EntityAdapterModel::addEntity(const EntityAdapter& entity)
	{
		beginInsertRows(QModelIndex(), rowCount(), rowCount());
		_entities << entity;
		endInsertRows();
	}

	int EntityAdapterModel::rowCount(const QModelIndex& parent) const
	{
		Q_UNUSED(parent);
		return _entities.count();
	}

	QVariant EntityAdapterModel::data(const QModelIndex& index, int role) const
	{
		if (index.row() < 0 || index.row() >= _entities.count())
			return QVariant();

		const auto& entity = _entities[index.row()];
		if (role == static_cast<int>(Roles::Name))
			return entity.name();
		else if (role == static_cast<int>(Roles::Components))
			return QVariant::fromValue(entity.components());
		else if (role == static_cast<int>(Roles::Visibility))
			return true;
		return QVariant();
	}

	QHash<int, QByteArray> EntityAdapterModel::roleNames() const
	{
		QHash<int, QByteArray> roles;
		roles[static_cast<int>(Roles::Name)] = "name";
		roles[static_cast<int>(Roles::Components)] = "components";
		roles[static_cast<int>(Roles::Visibility)] = "visibility";
		return roles;
	}
}
