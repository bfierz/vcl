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

import QtQuick 2.2
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4

import MeshViewerRendering 1.0

MeshView
{
	MouseArea
	{
		CheckBox
		{
			style: CheckBoxStyle
			{
				label: Text
				{
					color: "white"
					text: "Wireframe"
				}
			}
			checked: false

			onClicked:
			{
				renderer.renderWireframe = checked
			}
		}

		property var currentSceneEntity: Qt.point(0, 0)

		anchors.fill: parent
		acceptedButtons: Qt.LeftButton | Qt.RightButton
		onPressed:
		{
			if (mouse.button & Qt.LeftButton)
			{
				var ids = parent.selectObject(mouse.x, mouse.y)
				if (ids.x == 0)
				{
					parent.beginDrag(ids.y, mouse.x, mouse.y)
				}
				else
				{
					currentSceneEntity = ids;
					console.log("Object Id: ", currentSceneEntity.x, ", Primitive Id: ", currentSceneEntity.y)
				}
			}
			else if (mouse.button & Qt.RightButton)
			{
				scene.startRotate(mouse.x / width, mouse.y / height)
			}
		}
		onReleased:
		{
			if (mouse.button & Qt.LeftButton)
			{
				parent.endDrag()
			}
			else if (mouse.button & Qt.RightButton)
			{
				scene.endRotate()
				renderer.update()
			}
		}
		onPositionChanged:
		{
			if (pressedButtons & Qt.LeftButton)
			{
				parent.dragObject(mouse.x, mouse.y)
				if (currentSceneEntity.x > 0)
					parent.moveObjectToHandle(currentSceneEntity.x)
			}
			else if (pressedButtons & Qt.RightButton)
			{
				scene.rotate(mouse.x / width, mouse.y / height)
				renderer.update()
			}
		}
		onWheel:
		{
		}
	}
}
