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

class DemoImGuiApplication final : public ImGuiApplication
{
public:
	using ImGuiApplication::ImGuiApplication;

private:
	void EditTransform(float* cameraView, float* cameraProjection, float* matrix, bool editTransformDecomposition)
	{
		if (editTransformDecomposition)
		{
			if (ImGui::IsKeyPressed(90))
				mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
			if (ImGui::IsKeyPressed(69))
				mCurrentGizmoOperation = ImGuizmo::ROTATE;
			if (ImGui::IsKeyPressed(82)) // r Key
				mCurrentGizmoOperation = ImGuizmo::SCALE;
			if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
				mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
				mCurrentGizmoOperation = ImGuizmo::ROTATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
				mCurrentGizmoOperation = ImGuizmo::SCALE;
			float matrixTranslation[3], matrixRotation[3], matrixScale[3];
			ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation, matrixRotation, matrixScale);
			ImGui::InputFloat3("Tr", matrixTranslation);
			ImGui::InputFloat3("Rt", matrixRotation);
			ImGui::InputFloat3("Sc", matrixScale);
			ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix);

			if (mCurrentGizmoOperation != ImGuizmo::SCALE)
			{
				if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
					mCurrentGizmoMode = ImGuizmo::LOCAL;
				ImGui::SameLine();
				if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
					mCurrentGizmoMode = ImGuizmo::WORLD;
			}
			if (ImGui::IsKeyPressed(83))
				useSnap = !useSnap;
			ImGui::Checkbox("", &useSnap);
			ImGui::SameLine();

			switch (mCurrentGizmoOperation)
			{
			case ImGuizmo::TRANSLATE:
				ImGui::InputFloat3("Snap", &snap[0]);
				break;
			case ImGuizmo::ROTATE:
				ImGui::InputFloat("Angle Snap", &snap[0]);
				break;
			case ImGuizmo::SCALE:
				ImGui::InputFloat("Scale Snap", &snap[0]);
				break;
			}
			ImGui::Checkbox("Bound Sizing", &boundSizing);
			if (boundSizing)
			{
				ImGui::PushID(3);
				ImGui::Checkbox("", &boundSizingSnap);
				ImGui::SameLine();
				ImGui::InputFloat3("Snap", boundsSnap);
				ImGui::PopID();
			}
		}

		ImGuiIO& io = ImGui::GetIO();
		float viewManipulateRight = io.DisplaySize.x;
		float viewManipulateTop = 0;
		if (useWindow)
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
		ImGuizmo::DrawCubes(cameraView, cameraProjection, _objectTransforms[0].data(), gizmoCount);
		ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation, mCurrentGizmoMode, matrix, NULL, useSnap ? &snap[0] : NULL, boundSizing ? bounds : NULL, boundSizingSnap ? boundsSnap : NULL);

		ImGuizmo::ViewManipulate(cameraView, camDistance, ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128), 0x10101010);

		if (useWindow)
		{
			ImGui::End();
			ImGui::PopStyleColor(1);
		}
	}
	void updateFrame() override
	{
		ImGuiApplication::updateFrame();

		const ImGuiIO& io = ImGui::GetIO();
		if (isPerspective)
		{
			//Perspective(fov, io.DisplaySize.x / io.DisplaySize.y, 0.1f, 100.f, cameraProjection);
			const float full_fov_rad = 2.0f * fov * 3.141592f / 180.0f;
			_projection = _matrixFactory.createPerspectiveFov(0.1f, 100.f, io.DisplaySize.x / io.DisplaySize.y, full_fov_rad, Vcl::Graphics::Handedness::RightHanded);
		}
		else
		{
			float viewHeight = viewWidth * io.DisplaySize.y / io.DisplaySize.x;
			//OrthoGraphic(-viewWidth, viewWidth, -viewHeight, viewHeight, 1000.f, -1000.f, cameraProjection);
			_projection = _matrixFactory.createOrtho(2 * viewWidth, 2 * viewHeight, -1000.0f, 1000.0f, Vcl::Graphics::Handedness::RightHanded);
		}
		ImGuizmo::SetOrthographic(!isPerspective);
		ImGuizmo::BeginFrame();

		ImGui::SetNextWindowPos(ImVec2(1024, 100));
		ImGui::SetNextWindowSize(ImVec2(256, 256));

		// create a window and insert the inspector
		ImGui::SetNextWindowPos(ImVec2(10, 10));
		ImGui::SetNextWindowSize(ImVec2(320, 340));
		ImGui::Begin("Editor");
		if (ImGui::RadioButton("Full view", !useWindow)) useWindow = false;
		ImGui::SameLine();
		if (ImGui::RadioButton("Window", useWindow)) useWindow = true;

		ImGui::Text("Camera");
		if (ImGui::RadioButton("Perspective", isPerspective)) isPerspective = true;
		ImGui::SameLine();
		if (ImGui::RadioButton("Orthographic", !isPerspective)) isPerspective = false;
		if (isPerspective)
		{
			ImGui::SliderFloat("Fov", &fov, 20.f, 110.f);
		}
		else
		{
			ImGui::SliderFloat("Ortho width", &viewWidth, 1, 20);
		}
		bool viewDirty = false;
		viewDirty |= ImGui::SliderFloat("Distance", &camDistance, 1.f, 10.f);
		ImGui::SliderInt("Gizmo count", &gizmoCount, 1, 4);

		if (viewDirty || firstFrame)
		{
			Eigen::Vector3f eye{ cosf(camYAngle) * cosf(camXAngle) * camDistance, sinf(camXAngle) * camDistance, sinf(camYAngle) * cosf(camXAngle) * camDistance };
			Eigen::Vector3f at{ 0.f, 0.f, 0.f };
			Eigen::Vector3f up{ 0.f, 1.f, 0.f };
			Eigen::Matrix4f view = _matrixFactory.createLookAt(eye, -eye.normalized(), up);
			std::copy(view.data(), view.data() + 16, _view.data());
			firstFrame = false;
		}

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
		for (int matId = 0; matId < gizmoCount; matId++)
		{
			ImGuizmo::SetID(matId);

			EditTransform(_view.data(), _projection.data(), _objectTransforms[matId].data(), lastUsing == matId);
			if (ImGuizmo::IsUsing())
			{
				lastUsing = matId;
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
		rp_desc.RenderTargetAttachments[0].ClearColor = { clear_color.x, clear_color.y, clear_color.z, clear_color.w };
		rp_desc.RenderTargetAttachments[0].LoadOp = AttachmentLoadOp::Clear;
		rp_desc.DepthStencilTargetAttachment.Attachment = reinterpret_cast<void*>(dsv.ptr);
		rp_desc.DepthStencilTargetAttachment.ClearDepth = 1.0f;
		rp_desc.DepthStencilTargetAttachment.DepthLoadOp = AttachmentLoadOp::Clear;
		cmd_buffer->beginRenderPass(rp_desc);
		cmd_buffer->endRenderPass();

		ImGuiApplication::renderFrame(cmd_buffer, rtv, dsv);
	}

	Vcl::Graphics::Direct3D::MatrixFactory _matrixFactory;

	//! Projection matrix
	Eigen::Matrix4f _projection;

	//! Current view matrix
	std::array<float, 16> _view;

	//! Object states
	std::array<Eigen::Matrix4f, 4> _objectTransforms = {
		Eigen::Affine3f(Eigen::Translation3f(0, 0, 0)).matrix(),
		Eigen::Affine3f(Eigen::Translation3f(2, 0, 0)).matrix(),
		Eigen::Affine3f(Eigen::Translation3f(2, 0, 2)).matrix(),
		Eigen::Affine3f(Eigen::Translation3f(0, 0, 2)).matrix()
	};
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	ImGuizmo::OPERATION mCurrentGizmoOperation{ ImGuizmo::TRANSLATE };

	bool isPerspective = true;
	float fov = 27.f;
	float viewWidth = 10.f; // for orthographic
	float camYAngle = 165.f / 180.f * 3.14159f;
	float camXAngle = 32.f / 180.f * 3.14159f;

	// UI states
	bool useWindow = true;
	int gizmoCount = 1;
	float camDistance = 8.f;
	ImGuizmo::MODE mCurrentGizmoMode{ ImGuizmo::LOCAL };
	bool useSnap = false;
	float snap[3] = { 1.f, 1.f, 1.f };
	float bounds[6] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	float boundsSnap[3] = { 0.1f, 0.1f, 0.1f };
	bool boundSizing = false;
	bool boundSizingSnap = false;

	bool firstFrame = true;
	int lastUsing = 0;
};

int main(int argc, char** argv)
{
	DemoImGuiApplication app{"ImGui Demo"};
	return app.run();
}
