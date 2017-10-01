/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <memory>

// Qt
#include <QtCore/QAbstractNativeEventFilter>
#include <QtCore/QObject>
#include <QtQml/QJsEngine>
#include <QtQml/QQmlContext>

// VCL
#include <vcl/graphics/cameracontroller.h>
#include <vcl/hid/spacenavigatorhandler.h>

#include "../scene.h"

// Forward declaration
namespace Vcl { namespace HID { namespace Windows { class DeviceManager; }}}

namespace Editor
{
	class SpaceNavigatorController : public Vcl::Graphics::CameraController
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	public:
		void setRotationCenter(const Eigen::Vector3f& center) override;
		
		Eigen::Matrix4f objectTransformation() const override { return _objCurrTransformation; }

		void cameraControl(const std::array<float,6>& motion_data);
		void objectControl(const std::array<float, 6>& motion_data);
		void targetCameraControl(const std::array<float, 6>& motion_data);

	private:
		float relativeObjectSpeed(float targetDistance) const;

	private:
		//! Current rotation around the object
		Eigen::Matrix<float, 4, 4, Eigen::DontAlign> _objCurrTransformation{ Eigen::Matrix4f::Identity() };
		
		//! Current rotation center
		Eigen::Vector3f _objRotationCenter{ Eigen::Vector3f::Zero() };
	};

	class SpaceMouseController : public QObject, public QAbstractNativeEventFilter, public Vcl::HID::SpaceNavigatorHandler
	{
		Q_OBJECT
	public:
		static QObject* instance(QQmlEngine*, QJSEngine* engine);

	public:
		SpaceMouseController(QObject* parent = Q_NULLPTR);
		~SpaceMouseController();

		Q_INVOKABLE void attachTo(Scene* scene);
		
		//! Filter native events in order to process HID input events
		bool nativeEventFilter(const QByteArray& event_type, void* message, long* result) override;

	signals:
		void spaceMouseMoved();

	private:
		// Inherited via SpaceNavigatorHandler
		virtual void onSpaceMouseMove(const Vcl::HID::SpaceNavigator * device, std::array<float, 6> motion_data) override;
		virtual void onSpaceMouseKeyDown(const Vcl::HID::SpaceNavigator * device, unsigned int virtual_key) override;
		virtual void onSpaceMouseKeyUp(const Vcl::HID::SpaceNavigator * device, unsigned int virtual_key) override;

	private:
		std::unique_ptr<Vcl::HID::Windows::DeviceManager> _deviceManager;

		SpaceNavigatorController _controller;
	};
}
