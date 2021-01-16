/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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

#include "../common/imguiapp.h"

#include "../../3rdparty/imguizmo/ImGuizmo.h"

// C++ standard library
#include <utility>

// VCL
#include <vcl/graphics/d3d12/device.h>
#include <vcl/graphics/d3d12/swapchain.h>
#include <vcl/graphics/matrixfactory.h>

class DemoImGuizmoApplication final : public ImGuiApplication
{
public:
	DemoImGuizmoApplication(LPCSTR title)
	: ImGuiApplication(title)
	{
		computeViewMatrix();
	}

private:
	void computeViewMatrix()
	{
		Eigen::Vector3f eye{ cosf(cam_y_angle) * cosf(cam_x_angle) * _cam_distance, sinf(cam_x_angle) * _cam_distance, sinf(cam_y_angle) * cosf(cam_x_angle) * _cam_distance };
		Eigen::Vector3f at{ 0.f, 0.f, 0.f };
		Eigen::Vector3f up{ 0.f, 1.f, 0.f };
		Eigen::Matrix4f view = _matrixFactory.createLookAt(eye, -eye.normalized(), up);
		std::copy(view.data(), view.data() + 16, _view.data());
	}
	void EditTransform(float* cameraView, float* cameraProjection, float* matrix, bool editTransformDecomposition)
	{
		if (editTransformDecomposition)
		{
			if (ImGui::IsKeyPressed(90))
				_currentGizmoOperation = ImGuizmo::TRANSLATE;
			if (ImGui::IsKeyPressed(69))
				_currentGizmoOperation = ImGuizmo::ROTATE;
			if (ImGui::IsKeyPressed(82)) // r Key
				_currentGizmoOperation = ImGuizmo::SCALE;
			if (ImGui::RadioButton("Translate", _currentGizmoOperation == ImGuizmo::TRANSLATE))
				_currentGizmoOperation = ImGuizmo::TRANSLATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Rotate", _currentGizmoOperation == ImGuizmo::ROTATE))
				_currentGizmoOperation = ImGuizmo::ROTATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Scale", _currentGizmoOperation == ImGuizmo::SCALE))
				_currentGizmoOperation = ImGuizmo::SCALE;
			float matrixTranslation[3], matrixRotation[3], matrixScale[3];
			ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation, matrixRotation, matrixScale);
			ImGui::InputFloat3("Tr", matrixTranslation);
			ImGui::InputFloat3("Rt", matrixRotation);
			ImGui::InputFloat3("Sc", matrixScale);
			ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix);

			if (_currentGizmoOperation != ImGuizmo::SCALE)
			{
				if (ImGui::RadioButton("Local", _currentGizmoMode == ImGuizmo::LOCAL))
					_currentGizmoMode = ImGuizmo::LOCAL;
				ImGui::SameLine();
				if (ImGui::RadioButton("World", _currentGizmoMode == ImGuizmo::WORLD))
					_currentGizmoMode = ImGuizmo::WORLD;
			}
			if (ImGui::IsKeyPressed(83))
				_useSnap = !_useSnap;
			ImGui::Checkbox("", &_useSnap);
			ImGui::SameLine();

			switch (_currentGizmoOperation)
			{
			case ImGuizmo::TRANSLATE:
				ImGui::InputFloat3("Snap", &_snap[0]);
				break;
			case ImGuizmo::ROTATE:
				ImGui::InputFloat("Angle Snap", &_snap[0]);
				break;
			case ImGuizmo::SCALE:
				ImGui::InputFloat("Scale Snap", &_snap[0]);
				break;
			}
			ImGui::Checkbox("Bound Sizing", &_boundSizing);
			if (_boundSizing)
			{
				ImGui::PushID(3);
				ImGui::Checkbox("", &_boundSizingSnap);
				ImGui::SameLine();
				ImGui::InputFloat3("Snap", _boundsSnap.data());
				ImGui::PopID();
			}
		}

		ImGuiIO& io = ImGui::GetIO();
		float viewManipulateRight = io.DisplaySize.x;
		float viewManipulateTop = 0;
		if (_useWindow)
		{
			ImGui::SetNextWindowSize(ImVec2(800, 400));
			ImGui::SetNextWindowPos(ImVec2(400, 20));
			ImGui::PushStyleColor(ImGuiCol_WindowBg, (ImVec4)ImColor(0.35f, 0.3f, 0.3f));
			ImGui::Begin("Gizmo", 0, ImGuiWindowFlags_NoMove);
			ImGuizmo::SetDrawlist();
			float windowWidth = (float)ImGui::GetWindowWidth();
			float windowHeight = (float)ImGui::GetWindowHeight();
			ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, windowWidth, windowHeight);
			viewManipulateRight = ImGui::GetWindowPos().x + windowWidth;
			viewManipulateTop = ImGui::GetWindowPos().y;
		}
		else
		{
			ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
		}

		ImGuizmo::DrawGrid(cameraView, cameraProjection, Eigen::Matrix4f::Identity().eval().data(), 100.f);
		ImGuizmo::DrawCubes(cameraView, cameraProjection, _objectTransforms[0].data(), _gizmoCount);
		ImGuizmo::Manipulate(cameraView, cameraProjection, _currentGizmoOperation, _currentGizmoMode, matrix, nullptr, _useSnap ? _snap.data() : nullptr, _boundSizing ? _bounds.data() : nullptr, _boundSizingSnap ? _boundsSnap.data() : nullptr);

		ImGuizmo::ViewManipulate(cameraView, _cam_distance, ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128), 0x10101010);

		if (_useWindow)
		{
			ImGui::End();
			ImGui::PopStyleColor(1);
		}
	}
	void updateFrame() override
	{
		ImGuiApplication::updateFrame();

		const ImGuiIO& io = ImGui::GetIO();
		if (_usePerspective)
		{
			//Perspective(_fov, io.DisplaySize.x / io.DisplaySize.y, 0.1f, 100.f, cameraProjection);
			const float full_fov_rad = 2.0f * _fov * deg_to_rad;
			_projection = _matrixFactory.createPerspectiveFov(0.1f, 100.f, io.DisplaySize.x / io.DisplaySize.y, full_fov_rad, Vcl::Graphics::Handedness::RightHanded);
		}
		else
		{
			float viewHeight = _orthoViewWidth * io.DisplaySize.y / io.DisplaySize.x;
			//OrthoGraphic(-_orthoViewWidth, _orthoViewWidth, -viewHeight, viewHeight, 1000.f, -1000.f, cameraProjection);
			_projection = _matrixFactory.createOrtho(2 * _orthoViewWidth, 2 * viewHeight, -1000.0f, 1000.0f, Vcl::Graphics::Handedness::RightHanded);
		}
		ImGuizmo::SetOrthographic(!_usePerspective);
		ImGuizmo::BeginFrame();

		ImGui::SetNextWindowPos(ImVec2(1024, 100));
		ImGui::SetNextWindowSize(ImVec2(256, 256));

		// create a window and insert the inspector
		ImGui::SetNextWindowPos(ImVec2(10, 10));
		ImGui::SetNextWindowSize(ImVec2(320, 340));
		ImGui::Begin("Editor");
		if (ImGui::RadioButton("Full view", !_useWindow)) _useWindow = false;
		ImGui::SameLine();
		if (ImGui::RadioButton("Window", _useWindow)) _useWindow = true;

		ImGui::Text("Camera");
		if (ImGui::RadioButton("Perspective", _usePerspective)) _usePerspective = true;
		ImGui::SameLine();
		if (ImGui::RadioButton("Orthographic", !_usePerspective)) _usePerspective = false;
		if (_usePerspective)
		{
			ImGui::SliderFloat("Fov", &_fov, 20.f, 110.f);
		}
		else
		{
			ImGui::SliderFloat("Ortho width", &_orthoViewWidth, 1, 20);
		}
		if (ImGui::SliderFloat("Distance", &_cam_distance, 1.f, 10.f))
		{
			computeViewMatrix();
		}
		ImGui::SliderInt("Gizmo count", &_gizmoCount, 1, 4);

		ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);
		if (ImGuizmo::IsUsing())
		{
			ImGui::Text("Using gizmo");
		}
		else
		{
			ImGui::Text(ImGuizmo::IsOver() ? "Over gizmo" : "");
			ImGui::SameLine();
			ImGui::Text(ImGuizmo::IsOver(ImGuizmo::TRANSLATE) ? "Over translate gizmo" : "");
			ImGui::SameLine();
			ImGui::Text(ImGuizmo::IsOver(ImGuizmo::ROTATE) ? "Over rotate gizmo" : "");
			ImGui::SameLine();
			ImGui::Text(ImGuizmo::IsOver(ImGuizmo::SCALE) ? "Over scale gizmo" : "");
		}
		ImGui::Separator();
		for (int matId = 0; matId < _gizmoCount; matId++)
		{
			ImGuizmo::SetID(matId);

			EditTransform(_view.data(), _projection.data(), _objectTransforms[matId].data(), _lastUsingGizmo == matId);
			if (ImGuizmo::IsUsing())
			{
				_lastUsingGizmo = matId;
			}
		}

		ImGui::End();
	}
	void renderFrame(Vcl::Graphics::Runtime::D3D12::CommandBuffer* cmd_buffer, D3D12_CPU_DESCRIPTOR_HANDLE rtv, D3D12_CPU_DESCRIPTOR_HANDLE dsv) override
	{
		using namespace Vcl::Graphics::Runtime;

		RenderPassDescription rp_desc = {};
		rp_desc.RenderTargetAttachments.resize(1);
		rp_desc.RenderTargetAttachments[0].Attachment = reinterpret_cast<void*>(rtv.ptr);
		rp_desc.RenderTargetAttachments[0].ClearColor = { _clearColour.x, _clearColour.y, _clearColour.z, _clearColour.w };
		rp_desc.RenderTargetAttachments[0].LoadOp = AttachmentLoadOp::Clear;
		rp_desc.DepthStencilTargetAttachment.Attachment = reinterpret_cast<void*>(dsv.ptr);
		rp_desc.DepthStencilTargetAttachment.ClearDepth = 1.0f;
		rp_desc.DepthStencilTargetAttachment.DepthLoadOp = AttachmentLoadOp::Clear;
		cmd_buffer->beginRenderPass(rp_desc);
		cmd_buffer->endRenderPass();

		ImGuiApplication::renderFrame(cmd_buffer, rtv, dsv);
	}

	//! Degree to Rad conversion factor
	static constexpr float deg_to_rad = 3.141592f / 180.0f;
	//! Y-position of the camera
	static constexpr float cam_y_angle = 165.0f * deg_to_rad;
	//! X-position of the camera
	static constexpr float cam_x_angle = 32.0f * deg_to_rad;

	//! Camera half-opening angle
	float _fov = 27.0f;

	//! Distance to camera
	float _cam_distance = 8.f;

	//! View width for orthographic view
	float _orthoViewWidth = 10.f;

	//! Use perspective projection
	bool _usePerspective = true;

	//! View and projection matrix factory
	Vcl::Graphics::Direct3D::MatrixFactory _matrixFactory;

	//! Projection matrix
	Eigen::Matrix4f _projection;

	//! Current view matrix
	std::array<float, 16> _view;

	//! Number of objects and corresponding gizmos
	int _gizmoCount = 1;

	//! Object states
	std::array<Eigen::Matrix4f, 4> _objectTransforms = {
		Eigen::Affine3f(Eigen::Translation3f(0, 0, 0)).matrix(),
		Eigen::Affine3f(Eigen::Translation3f(2, 0, 0)).matrix(),
		Eigen::Affine3f(Eigen::Translation3f(2, 0, 2)).matrix(),
		Eigen::Affine3f(Eigen::Translation3f(0, 0, 2)).matrix()
	};

	//! Background colour
	const ImVec4 _clearColour = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	//! Windowed mode
	bool _useWindow = true;

	//! Current Gizmo operation
	ImGuizmo::OPERATION _currentGizmoOperation{ ImGuizmo::TRANSLATE };

	//! Current Gizmo operation mode
	ImGuizmo::MODE _currentGizmoMode{ ImGuizmo::LOCAL };

	//! Last edited Gizmo
	int _lastUsingGizmo = 0;

	//! Snapping configuration
	bool _useSnap = false;
	std::array<float, 3> _snap = { 1.f, 1.f, 1.f };
	std::array<float, 6> _bounds = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	std::array<float, 3> _boundsSnap = { 0.1f, 0.1f, 0.1f };
	bool _boundSizing = false;
	bool _boundSizingSnap = false;
};

int main(int argc, char** argv)
{
	DemoImGuizmoApplication app{"ImGui Demo"};
	return app.run();
}
