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
#include <vcl/config/opengl.h>

// C++ standard library

// Qt
#include <QtGui/QGuiApplication>
#include <QtQml/QQmlApplicationEngine>
#include <QtQml/QQmlContext>
#include <QtQuick/QQuickWindow>

// VCL

#include "meshview.h"
#include "scene.h"

// Force the use of the NVIDIA GPU in an Optimius system
extern "C"
{
	_declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
}

int main(int argc, char **argv)
{
	QGuiApplication app(argc, argv);
	app.setOrganizationName("");
	app.setOrganizationDomain("");

	// Make types accessible in QML
	qmlRegisterType<MeshView>("MeshViewerRendering", 1, 0, "MeshView");
	qmlRegisterType<Scene>("MeshViewerRendering", 1, 0, "Scene");

	// Configure the default OpenGL format
	auto gl_fmt = QSurfaceFormat::defaultFormat();
	gl_fmt.setProfile(QSurfaceFormat::CoreProfile);
	gl_fmt.setVersion(4, 5);
	gl_fmt.setOptions(QSurfaceFormat::DebugContext | QSurfaceFormat::ResetNotification);
	QSurfaceFormat::setDefaultFormat(gl_fmt);

	QQmlApplicationEngine engine;

	Scene scene;
	engine.rootContext()->setContextProperty("scene", &scene);

	engine.load(QUrl("qrc:///main.qml"));

	QObject *topLevel = engine.rootObjects().value(0);
	QQuickWindow *window = qobject_cast<QQuickWindow*>(topLevel);
	if (!window)
	{
		qWarning("Error: Your root item has to be a Window.");
		return -1;	
	}
	window->show();

	return app.exec();
}
