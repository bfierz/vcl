/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
#include <array>

// Abseil
#include <absl/container/inlined_vector.h>

//  VCL
#include <vcl/core/memory/smart_ptr.h>
#include <vcl/graphics/runtime/resource/texture.h>

namespace Vcl { namespace Graphics { namespace Runtime {
	enum class ComparisonFunction
	{
		Never = 1,
		Less = 2,
		Equal = 3,
		LessEqual = 4,
		Greater = 5,
		NotEqual = 6,
		GreaterEqual = 7,
		Always = 8
	};

	enum class AttachmentLoadOp
	{
		DontCare,
		Clear,
		Load
	};

	enum class AttachmentStoreOp
	{
		Store,
		Clear
	};

	struct RenderTargetAttachmentDescription
	{
		union
		{
			void* Attachment = nullptr;
			Core::ref_ptr<Texture> View;
		};
		AttachmentLoadOp LoadOp = AttachmentLoadOp::DontCare;
		AttachmentStoreOp StoreOp = AttachmentStoreOp::Store;
		std::array<float, 4> ClearColor = { 0, 0, 0, 0 };
	};

	struct DepthStencilAttachmentTargetDescription
	{
		union
		{
			void* Attachment = nullptr;
			Core::ref_ptr<Texture> View;
		};
		AttachmentLoadOp DepthLoadOp = AttachmentLoadOp::DontCare;
		AttachmentStoreOp DepthStoreOp = AttachmentStoreOp::Store;
		float ClearDepth = 1.0f;
		AttachmentLoadOp StencilLoadOp = AttachmentLoadOp::DontCare;
		AttachmentStoreOp StencilStoreOp = AttachmentStoreOp::Store;
		uint32_t ClearStencil = 0;
	};

	struct RenderPassDescription
	{
		absl::InlinedVector<RenderTargetAttachmentDescription, 8> RenderTargetAttachments;
		DepthStencilAttachmentTargetDescription DepthStencilTargetAttachment;
	};
}}}
