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
#include "spacemousecontroller.h"

// Qt
#include <QtCore/QCoreApplication>
#include <QtGui/QInputEvent>

// VCL
#include <vcl/graphics/cameracontroller.h>
#include <vcl/hid/windows/hid.h>
#include <vcl/hid/spacenavigator.h>

namespace Editor
{
	void SpaceNavigatorController::setRotationCenter(const Eigen::Vector3f& center)
	{
		_objRotationCenter = center;
	}
	
	void SpaceNavigatorController::objectControl(const std::array<float, 6>& motion_data)
	{
		using namespace Eigen;

		if (camera() == nullptr)
			return;

		// Motion is in camera coordinate system and is normally applied
		// to the camera. To move the object in the direction of the forces 
		// applied to the cap, data needs to be negated.
		Vector3f translation(-motion_data[0], -motion_data[2], -motion_data[1]);
		Vector3f rotation(   -motion_data[3], -motion_data[5], -motion_data[4]);

		// Scale translation and rotation for the FOV
		translation.x() *= tan(camera()->fieldOfView() / 2.0f);
		translation.y() *= tan(camera()->fieldOfView() / 2.0f);
		rotation.x() *= tan(camera()->fieldOfView() / 2.0f);
		rotation.y() *= tan(camera()->fieldOfView() / 2.0f);
		
		// Camera transformation/position
		Affine3f world2camera(camera()->view());
		Affine3f camera2world(world2camera.inverse());
		Vector3f camera_pos = camera()->position();

		Affine3f model2world(_objCurrTransformation);
		Vector3f pivot_wc = model2world * _objRotationCenter;

		Vector3f camera2Pivot_wc = pivot_wc - camera_pos;
		Vector3f camera2Pivot_cc = world2camera.linear() * camera2Pivot_wc;

		Affine3f camera2Pivot;
		camera2Pivot.setIdentity();
		camera2Pivot.translationExt() = camera2Pivot_cc;

		// Use distance to pivot to convert to longitudinal displacement
		translation *= relativeObjectSpeed(camera2Pivot_cc.norm());

#if _LOCK_HORIZON_INPUT_ALGORITHM
		if (m_view.Standard3dmouse()->IsLockHorizon())
			rotation = lockHorizon(rotation);
#endif

		Affine3f transform;
		transform.setIdentity();

		// Set translation
		//transform.translationExt() = translation;

		// Set rotation
		float angle = rotation.norm();
		if (angle > 1.0e-5f)
		{
			Vector3f axis = rotation / angle;
			transform.linearExt() = AngleAxisf(angle, axis).toRotationMatrix();
		}

		//transform = camera2Pivot * transform * camera2Pivot.inverse();

		// Perform the incremental rotation and translation due to 3D mouse
		_objCurrTransformation = transform * _objCurrTransformation;

#if _LOCK_HORIZON_OUTPUT_ALGORITHM
		if (m_view.Standard3dmouse()->IsLockHorizon())
			levelHorizon(pivotPosWC, camera2worldTM);
#endif
			  
		//float distance = (camera()->position() - camera()->target()).norm();
		//Vector3f p = camera2world.linear() * (camera()->position() - camera()->target());
		//Vector3f up = camera2world.linear() * camera()->up();
		
		//camera()->setPosition(camera()->position() + p + translation);
		//camera()->setTarget(camera()->target() + translation);
		//camera()->setUp(up.normalized());

		// Handle Parallel projections
		//if (m_view.Projection != ePerspective)
		//   fovZoom(translation.z);
	}
	
   float SpaceNavigatorController::relativeObjectSpeed(float target_distance) const
   {
		// Handle Parallel projections
		//if (m_view.Projection != ePerspective)
		//	targetDistance = m_view.TargetDistance;

		if (target_distance < 0.0f)
			target_distance = -target_distance;

		if (target_distance < camera()->nearPlane())
			target_distance = camera()->nearPlane();

		return target_distance;
   }
	
	QObject* SpaceMouseController::instance(QQmlEngine*, QJSEngine* engine)
	{
		return new SpaceMouseController(engine);
	}
	
	SpaceMouseController::SpaceMouseController(QObject *parent)
	: QObject(parent)
	, _deviceManager(std::make_unique<Vcl::HID::Windows::DeviceManager>())
	{
		_deviceManager->registerDevices(Vcl::HID::DeviceType::MultiAxisController, nullptr);
		
		qApp->installNativeEventFilter(this);
	}
	SpaceMouseController::~SpaceMouseController()
	{
		Vcl::HID::SpaceNavigator::unregisterHandler(this);
	}

	void SpaceMouseController::attachTo(Scene* scene)
	{
		if (!scene)
			return;

		Vcl::HID::SpaceNavigator::registerHandler(this);

		// Grab control of the camera
		scene->setCameraController(&_controller);
		_controller.setCamera(scene->camera());
	}

	bool SpaceMouseController::nativeEventFilter(const QByteArray& event_type, void* message, long* result)
	{
		if (event_type == "windows_generic_MSG")
		{
			MSG* msg = reinterpret_cast<MSG*>(message);
			if (msg->message == WM_INPUT)
			{
				if (_deviceManager->processInput(msg->hwnd, WM_INPUT, msg->wParam, msg->lParam))
				{
					return true;
				}
			}
		}
		return false;
	}
	void SpaceMouseController::onSpaceMouseMove(const Vcl::HID::SpaceNavigator * device, std::array<float, 6> motion_data)
	{
		_controller.objectControl(motion_data);

		// Notify users
		spaceMouseMoved();
	}
	void SpaceMouseController::onSpaceMouseKeyDown(const Vcl::HID::SpaceNavigator * device, unsigned int virtual_key)
	{
	}
	void SpaceMouseController::onSpaceMouseKeyUp(const Vcl::HID::SpaceNavigator * device, unsigned int virtual_key)
	{
	}
}
