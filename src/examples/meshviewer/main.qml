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

import QtQuick 2.2
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2
import QtQuick.Window 2.1

import MeshViewerRendering 1.0

ApplicationWindow
{
    width: 800
    height: 800

	OpenMeshDialog
	{
		id: openMeshDialog
		
		onAccepted:
		{
			console.log("Accepted: " + fileUrls)
			scene.loadMesh(openMeshDialog.fileUrl)
			renderer.update()
		}
		onRejected: { console.log("Rejected") }
	}

	CreateBarDialog
	{
		id: createBarDialog
		
		onAccepted:
		{
			scene.createBar(createBarDialog.xResolution, createBarDialog.yResolution, createBarDialog.zResolution)
			renderer.update()
		}
	}

	menuBar: MenuBar
	{
        Menu
		{
            title: "File"
            MenuItem
			{
				text: "Open..."
				onTriggered: { openMeshDialog.open() }
			}
            MenuItem
			{
				text: "Exit"
				onTriggered: { Qt.quit() }
			}
        }

        Menu
		{
            title: "Create"
			Menu
			{
				title: "Triangle Mesh"
				MenuItem
				{
					text: "Sphere"
					onTriggered:
					{
						scene.createSurfaceSphere()
						renderer.update()
					}
				}
			}
			Menu
			{
				title: "Tetra Mesh"
				MenuItem
				{
					text: "Bar"
					onTriggered: { createBarDialog.open() }
				}
			}
        }
    }

	MeshView
	{
        id: renderer
        anchors.fill: parent
        anchors.margins: 10
		
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


		MouseArea
		{
			anchors.fill: parent
			acceptedButtons: Qt.LeftButton | Qt.MiddleButton
			onPressed:
			{
				if (mouse.button & Qt.LeftButton)
				{
					parent.selectObject(mouse.x, mouse.y)
				}
				else if (mouse.button & Qt.MiddleButton)
				{
					scene.startRotate(mouse.x / width, mouse.y / height)
				}
			}
			onReleased:
			{
				if (mouse.button & Qt.MiddleButton)
				{
					scene.endRotate()
					renderer.update()
				}
			}
			onPositionChanged:
			{
				scene.rotate(mouse.x / width, mouse.y / height)
				renderer.update()
			}
			onWheel:
			{
			}
		}
	}

	Component.onCompleted: { renderer.scene = scene }
}
