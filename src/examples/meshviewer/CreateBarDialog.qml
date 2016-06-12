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

import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.2
import QtQuick.Window 2.2

Dialog
{
	title: "Create a bar mesh"
	modality: Qt.WindowModal
	standardButtons: StandardButton.Ok | StandardButton.Cancel
	visible: false

	property int xResolution: xResolutionInput.text
	property int yResolution: yResolutionInput.text
	property int zResolution: zResolutionInput.text

	ColumnLayout
	{
		width: parent.width
		
		Label
		{
			text: "Mesh type"
			Layout.alignment: Qt.AlignBaseline | Qt.AlignLeft
		}

		RowLayout
		{
			Layout.alignment: Qt.AlignHCenter
		
			Label
			{
				text: "x"
			}

			TextField
			{
				id: xResolutionInput
				Layout.alignment: Qt.AlignBaseline
				Layout.fillWidth: true
				validator: IntValidator {bottom: 1; top: 512;}
				text: "1"
			}
			
			Label
			{
				text: "y"
			}

			TextField
			{
				id: yResolutionInput
				Layout.alignment: Qt.AlignBaseline
				Layout.fillWidth: true
				validator: IntValidator {bottom: 1; top: 512;}
				text: "1"
			}
			
			Label
			{
				text: "z"
			}

			TextField
			{
				id: zResolutionInput
				Layout.alignment: Qt.AlignBaseline
				Layout.fillWidth: true
				validator: IntValidator {bottom: 1; top: 512;}
				text: "1"
			}
		}
	}
}
