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
#include <vcl/graphics/d3d12/d3d.h>

// VCL library
#include <vcl/core/contract.h>

namespace Vcl { namespace Graphics { namespace D3D12 {
	/*AnyRenderType D3D::toRenderType(SurfaceFormat fmt)
	{
		switch (fmt)
		{
		case SurfaceFormat::R32G32B32A32_FLOAT  : return RenderType<Eigen::Vector4f>();
		//case SurfaceFormat::R32G32B32A32_UINT   : return RenderType<Eigen::Vector4f>();
		case SurfaceFormat::R32G32B32A32_SINT   : return RenderType<Eigen::Vector4i>();
		//case SurfaceFormat::R16G16B16A16_FLOAT  : return RenderType<Eigen::Vector4f>();
		//case SurfaceFormat::R16G16B16A16_UNORM  : gl_format = GL_RGBA16; break;
		//case SurfaceFormat::R16G16B16A16_UINT   : return RenderType<Eigen::Vector4f>();
		//case SurfaceFormat::R16G16B16A16_SNORM  : gl_format = GL_RGBA16_SNORM; break;
		//case SurfaceFormat::R16G16B16A16_SINT   : return RenderType<Eigen::Vector4f>();
		case SurfaceFormat::R32G32B32_FLOAT     : return RenderType<Eigen::Vector3f>();
		//case SurfaceFormat::R32G32B32_UINT      : return RenderType<Eigen::Vector4f>();
		case SurfaceFormat::R32G32B32_SINT      : return RenderType<Eigen::Vector3i>();
		case SurfaceFormat::R32G32_FLOAT        : return RenderType<Eigen::Vector2f>();
		//case SurfaceFormat::R32G32_UINT         : return RenderType<Eigen::Vector4f>();
		case SurfaceFormat::R32G32_SINT         : return RenderType<Eigen::Vector2i>();
		//case SurfaceFormat::D32_FLOAT_S8X24_UINT: gl_format = GL_DEPTH32F_STENCIL8; break;
		//case SurfaceFormat::R10G10B10A2_UNORM   : gl_format = GL_RGB10_A2; break;
		//case SurfaceFormat::R10G10B10A2_UINT    : gl_format = GL_RGB10_A2UI; break;
		//case SurfaceFormat::R11G11B10_FLOAT     : gl_format = GL_R11F_G11F_B10F; break;
		//case SurfaceFormat::R8G8B8A8_UNORM      : gl_format = GL_RGBA8; break;
		//case SurfaceFormat::R8G8B8A8_UNORM_SRGB : gl_format = GL_SRGB8_ALPHA8; break;
		//case SurfaceFormat::R8G8B8A8_UINT       : gl_format = GL_RGBA8UI; break;
		//case SurfaceFormat::R8G8B8A8_SNORM      : gl_format = GL_RGBA8_SNORM; break;
		//case SurfaceFormat::R8G8B8A8_SINT       : gl_format = GL_RGBA8I; break;
		//case SurfaceFormat::R16G16_FLOAT        : gl_format = GL_RG16F; break;
		//case SurfaceFormat::R16G16_UNORM        : gl_format = GL_RG16; break;
		//case SurfaceFormat::R16G16_UINT         : gl_format = GL_RG16UI; break;
		//case SurfaceFormat::R16G16_SNORM        : gl_format = GL_RG16_SNORM; break;
		//case SurfaceFormat::R16G16_SINT         : gl_format = GL_RG16I; break;
		//case SurfaceFormat::D32_FLOAT           : gl_format = GL_DEPTH_COMPONENT32F; break;
		//case SurfaceFormat::R32_FLOAT           : gl_format = GL_R32F; break;
		//case SurfaceFormat::R32_UINT            : gl_format = GL_R32UI; break;
		//case SurfaceFormat::R32_SINT            : gl_format = GL_R32I; break;
		//case SurfaceFormat::D24_UNORM_S8_UINT   : gl_format = GL_DEPTH24_STENCIL8; break;
		//case SurfaceFormat::R8G8_UNORM          : gl_format = GL_RG8; break;
		//case SurfaceFormat::R8G8_UINT           : gl_format = GL_RG8UI; break;
		//case SurfaceFormat::R8G8_SNORM          : gl_format = GL_RG8_SNORM; break;
		//case SurfaceFormat::R8G8_SINT           : gl_format = GL_RG8I; break;
		//case SurfaceFormat::R16_FLOAT           : gl_format = GL_R16F; break;
		//case SurfaceFormat::D16_UNORM           : gl_format = GL_DEPTH_COMPONENT16; break;
		//case SurfaceFormat::R16_UNORM           : gl_format = GL_R16; break;
		//case SurfaceFormat::R16_UINT            : gl_format = GL_R16UI; break;
		//case SurfaceFormat::R16_SNORM           : gl_format = GL_R16_SNORM; break;
		//case SurfaceFormat::R16_SINT            : gl_format = GL_R16I; break;
		//case SurfaceFormat::R8_UNORM            : gl_format = GL_R8; break;
		//case SurfaceFormat::R8_UINT             : gl_format = GL_R8UI; break;
		//case SurfaceFormat::R8_SNORM            : gl_format = GL_R8_SNORM; break;
		//case SurfaceFormat::R8_SINT             : gl_format = GL_R8I; break;
		default: VclDebugError("Unsupported colour format.");
		};

		return RenderType<void>();
	}*/

	DXGI_FORMAT D3D::toD3Denum(SurfaceFormat type)
	{
		DXGI_FORMAT d3d_format = DXGI_FORMAT_UNKNOWN;

		switch (type)
		{
		case SurfaceFormat::Unknown             : d3d_format = DXGI_FORMAT_UNKNOWN; break;
		case SurfaceFormat::R32G32B32A32_FLOAT  : d3d_format = DXGI_FORMAT_R32G32B32A32_FLOAT; break;
		case SurfaceFormat::R32G32B32A32_UINT   : d3d_format = DXGI_FORMAT_R32G32B32A32_UINT; break;
		case SurfaceFormat::R32G32B32A32_SINT   : d3d_format = DXGI_FORMAT_R32G32B32A32_SINT; break;
		case SurfaceFormat::R16G16B16A16_FLOAT  : d3d_format = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
		case SurfaceFormat::R16G16B16A16_UNORM  : d3d_format = DXGI_FORMAT_R16G16B16A16_UNORM; break;
		case SurfaceFormat::R16G16B16A16_UINT   : d3d_format = DXGI_FORMAT_R16G16B16A16_UINT; break;
		case SurfaceFormat::R16G16B16A16_SNORM  : d3d_format = DXGI_FORMAT_R16G16B16A16_SNORM; break;
		case SurfaceFormat::R16G16B16A16_SINT   : d3d_format = DXGI_FORMAT_R16G16B16A16_SINT; break;
		case SurfaceFormat::R32G32B32_FLOAT     : d3d_format = DXGI_FORMAT_R32G32B32_FLOAT; break;
		case SurfaceFormat::R32G32B32_UINT      : d3d_format = DXGI_FORMAT_R32G32B32_UINT; break;
		case SurfaceFormat::R32G32B32_SINT      : d3d_format = DXGI_FORMAT_R32G32B32_SINT; break;
		case SurfaceFormat::R32G32_FLOAT        : d3d_format = DXGI_FORMAT_R32G32_FLOAT; break;
		case SurfaceFormat::R32G32_UINT         : d3d_format = DXGI_FORMAT_R32G32_UINT; break;
		case SurfaceFormat::R32G32_SINT         : d3d_format = DXGI_FORMAT_R32G32_SINT; break;
		case SurfaceFormat::D32_FLOAT_S8X24_UINT: d3d_format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT; break;
		case SurfaceFormat::R10G10B10A2_UNORM   : d3d_format = DXGI_FORMAT_R10G10B10A2_UNORM; break;
		case SurfaceFormat::R10G10B10A2_UINT    : d3d_format = DXGI_FORMAT_R10G10B10A2_UINT; break;    
		case SurfaceFormat::R11G11B10_FLOAT     : d3d_format = DXGI_FORMAT_R11G11B10_FLOAT; break;     
		case SurfaceFormat::R8G8B8A8_UNORM      : d3d_format = DXGI_FORMAT_R8G8B8A8_UNORM; break;      
		case SurfaceFormat::R8G8B8A8_UNORM_SRGB : d3d_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; break;
		case SurfaceFormat::R8G8B8A8_UINT       : d3d_format = DXGI_FORMAT_R8G8B8A8_UINT; break;       
		case SurfaceFormat::R8G8B8A8_SNORM      : d3d_format = DXGI_FORMAT_R8G8B8A8_SNORM; break;      
		case SurfaceFormat::R8G8B8A8_SINT       : d3d_format = DXGI_FORMAT_R8G8B8A8_SINT; break;       
		case SurfaceFormat::R16G16_FLOAT        : d3d_format = DXGI_FORMAT_R16G16_FLOAT; break;        
		case SurfaceFormat::R16G16_UNORM        : d3d_format = DXGI_FORMAT_R16G16_UNORM; break;        
		case SurfaceFormat::R16G16_UINT         : d3d_format = DXGI_FORMAT_R16G16_UINT; break;         
		case SurfaceFormat::R16G16_SNORM        : d3d_format = DXGI_FORMAT_R16G16_SNORM; break;        
		case SurfaceFormat::R16G16_SINT         : d3d_format = DXGI_FORMAT_R16G16_SINT; break;         
		case SurfaceFormat::D32_FLOAT           : d3d_format = DXGI_FORMAT_D32_FLOAT; break;           
		case SurfaceFormat::R32_FLOAT           : d3d_format = DXGI_FORMAT_R32_FLOAT; break;           
		case SurfaceFormat::R32_UINT            : d3d_format = DXGI_FORMAT_R32_UINT; break;            
		case SurfaceFormat::R32_SINT            : d3d_format = DXGI_FORMAT_R32_SINT; break;            
		case SurfaceFormat::D24_UNORM_S8_UINT   : d3d_format = DXGI_FORMAT_D24_UNORM_S8_UINT; break;   
		case SurfaceFormat::R8G8_UNORM          : d3d_format = DXGI_FORMAT_R8G8_UNORM; break;          
		case SurfaceFormat::R8G8_UINT           : d3d_format = DXGI_FORMAT_R8G8_UINT; break;           
		case SurfaceFormat::R8G8_SNORM          : d3d_format = DXGI_FORMAT_R8G8_SNORM; break;          
		case SurfaceFormat::R8G8_SINT           : d3d_format = DXGI_FORMAT_R8G8_SINT; break;           
		case SurfaceFormat::R16_FLOAT           : d3d_format = DXGI_FORMAT_R16_FLOAT; break;           
		case SurfaceFormat::D16_UNORM           : d3d_format = DXGI_FORMAT_D16_UNORM; break;           
		case SurfaceFormat::R16_UNORM           : d3d_format = DXGI_FORMAT_R16_UNORM; break;           
		case SurfaceFormat::R16_UINT            : d3d_format = DXGI_FORMAT_R16_UINT; break;            
		case SurfaceFormat::R16_SNORM           : d3d_format = DXGI_FORMAT_R16_SNORM; break;           
		case SurfaceFormat::R16_SINT            : d3d_format = DXGI_FORMAT_R16_SINT; break;            
		case SurfaceFormat::R8_UNORM            : d3d_format = DXGI_FORMAT_R8_UNORM; break;           
		case SurfaceFormat::R8_UINT             : d3d_format = DXGI_FORMAT_R8_UINT; break;             
		case SurfaceFormat::R8_SNORM            : d3d_format = DXGI_FORMAT_R8_SNORM; break;
		case SurfaceFormat::R8_SINT             : d3d_format = DXGI_FORMAT_R8_SINT; break;
		default: VclDebugError("Unsupported colour format.");
		}
		
		return d3d_format;
	}

	/*D3D11_RTV_DIMENSION D3D::toD3Denum(RenderTargetViewDimension::RenderTargetViewDimension dim)
	{
		switch (dim)
		{
		case RenderTargetViewDimension::Unknown            : return D3D11_RTV_DIMENSION_UNKNOWN;
		case RenderTargetViewDimension::Buffer             : return D3D11_RTV_DIMENSION_BUFFER;
		case RenderTargetViewDimension::Texture1D          : return D3D11_RTV_DIMENSION_TEXTURE1D;
		case RenderTargetViewDimension::Texture1DArray     : return D3D11_RTV_DIMENSION_TEXTURE1DARRAY;
		case RenderTargetViewDimension::Texture2D          : return D3D11_RTV_DIMENSION_TEXTURE2D;
		case RenderTargetViewDimension::Texture2DArray     : return D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
		case RenderTargetViewDimension::Texture2DMS        : return D3D11_RTV_DIMENSION_TEXTURE2DMS;
		case RenderTargetViewDimension::Texture2DMSArray   : return D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
		case RenderTargetViewDimension::Texture3D          : return D3D11_RTV_DIMENSION_TEXTURE3D;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}
		
		DebugError("Invalid enum");
		return D3D11_RTV_DIMENSION_UNKNOWN;
	}

	D3D11_DSV_DIMENSION D3D::toD3Denum(DepthStencilViewDimension::DepthStencilViewDimension dim)
	{
		switch (dim)
		{
		case DepthStencilViewDimension::Unknown            : return D3D11_DSV_DIMENSION_UNKNOWN;
		case DepthStencilViewDimension::Texture1D          : return D3D11_DSV_DIMENSION_TEXTURE1D;
		case DepthStencilViewDimension::Texture1DArray     : return D3D11_DSV_DIMENSION_TEXTURE1DARRAY;
		case DepthStencilViewDimension::Texture2D          : return D3D11_DSV_DIMENSION_TEXTURE2D;
		case DepthStencilViewDimension::Texture2DArray     : return D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
		case DepthStencilViewDimension::Texture2DMS        : return D3D11_DSV_DIMENSION_TEXTURE2DMS;
		case DepthStencilViewDimension::Texture2DMSArray   : return D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}
		
		DebugError("Invalid enum");
		return D3D11_DSV_DIMENSION_UNKNOWN;
	}

	D3D11_SRV_DIMENSION D3D::toD3Denum(ShaderResourceViewDimension::ShaderResourceViewDimension dim)
	{
		switch (dim)
		{
		case ShaderResourceViewDimension::Unknown            : return D3D11_SRV_DIMENSION_UNKNOWN;
		case ShaderResourceViewDimension::Buffer             : return D3D11_SRV_DIMENSION_BUFFER;
		case ShaderResourceViewDimension::Texture1D          : return D3D11_SRV_DIMENSION_TEXTURE1D;
		case ShaderResourceViewDimension::Texture1DArray     : return D3D11_SRV_DIMENSION_TEXTURE1DARRAY;
		case ShaderResourceViewDimension::Texture2D          : return D3D11_SRV_DIMENSION_TEXTURE2D;
		case ShaderResourceViewDimension::Texture2DArray     : return D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
		case ShaderResourceViewDimension::Texture2DMS        : return D3D11_SRV_DIMENSION_TEXTURE2DMS;
		case ShaderResourceViewDimension::Texture2DMSArray   : return D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
		case ShaderResourceViewDimension::Texture3D          : return D3D11_SRV_DIMENSION_TEXTURE3D;
		case ShaderResourceViewDimension::TextureCube        : return D3D11_SRV_DIMENSION_TEXTURECUBE;
		case ShaderResourceViewDimension::TextureCubeArray   : return D3D11_SRV_DIMENSION_TEXTURECUBEARRAY;
//		case ShaderResourceViewDimension::BufferEx           : return D3D11_SRV_DIMENSION_BUFFEREX;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}
		
		DebugError("Invalid enum");
		return D3D11_SRV_DIMENSION_UNKNOWN;
	}

	D3D11_QUERY D3D::toD3Denum(VCL_ENUM(QueryType) query)
	{
		switch (query)
		{
		case QueryType::Event                            : return D3D11_QUERY_EVENT;
        case QueryType::Occlusion                        : return D3D11_QUERY_OCCLUSION;
        case QueryType::Timestamp                        : return D3D11_QUERY_TIMESTAMP;
        case QueryType::TimestampDisjoint                : return D3D11_QUERY_TIMESTAMP_DISJOINT;
        case QueryType::PipelineStatistics               : return D3D11_QUERY_PIPELINE_STATISTICS;
        case QueryType::OcclusionPredicate               : return D3D11_QUERY_OCCLUSION_PREDICATE;
        case QueryType::StreamOutStatistics              : return D3D11_QUERY_SO_STATISTICS;
        case QueryType::StreamOutOverflowPredicate       : return D3D11_QUERY_SO_OVERFLOW_PREDICATE;
        case QueryType::StreamOutStatisticsStream0       : return D3D11_QUERY_SO_STATISTICS_STREAM0;
        case QueryType::StreamOutOverflowPredicateStream0: return D3D11_QUERY_SO_OVERFLOW_PREDICATE_STREAM0;
        case QueryType::StreamOutStatistics_stream1      : return D3D11_QUERY_SO_STATISTICS_STREAM1;
        case QueryType::StreamOutOverflowPredicateStream1: return D3D11_QUERY_SO_OVERFLOW_PREDICATE_STREAM1;
        case QueryType::StreamOutStatistics_stream2      : return D3D11_QUERY_SO_STATISTICS_STREAM2;
        case QueryType::StreamOutOverflowPredicateStream2: return D3D11_QUERY_SO_OVERFLOW_PREDICATE_STREAM2;
        case QueryType::StreamOutStatistics_stream3      : return D3D11_QUERY_SO_STATISTICS_STREAM3;
		case QueryType::StreamOutOverflowPredicateStream3: return D3D11_QUERY_SO_OVERFLOW_PREDICATE_STREAM3;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}
		
		DebugError("Invalid enum");
		return D3D11_QUERY_EVENT;
	}

	D3D11_BLEND D3D::toD3Denum(Blend::Blend blend)
	{
		switch (blend)
		{
		case Blend::Zero            : return D3D11_BLEND_ZERO;
		case Blend::One             : return D3D11_BLEND_ONE;
		case Blend::SrcColour       : return D3D11_BLEND_SRC_COLOR;
		case Blend::InvSrcColour    : return D3D11_BLEND_INV_SRC_COLOR;
		case Blend::SrcAlpha        : return D3D11_BLEND_SRC_ALPHA;
		case Blend::InvSrcAlpha     : return D3D11_BLEND_INV_SRC_ALPHA;
		case Blend::DestAlpha       : return D3D11_BLEND_DEST_ALPHA;
		case Blend::InvDestAlpha    : return D3D11_BLEND_INV_DEST_ALPHA;
		case Blend::DestColour      : return D3D11_BLEND_DEST_COLOR;
		case Blend::InvDestColour   : return D3D11_BLEND_INV_DEST_COLOR;
		case Blend::SrcAlphaSat     : return D3D11_BLEND_SRC_ALPHA_SAT;
		case Blend::BlendFactor     : return D3D11_BLEND_BLEND_FACTOR;
		case Blend::InvBlendFactor	: return D3D11_BLEND_INV_BLEND_FACTOR;
		case Blend::Src1Colour      : return D3D11_BLEND_SRC1_COLOR;
		case Blend::InvSrc1Colour   : return D3D11_BLEND_INV_SRC1_COLOR;
		case Blend::Src1Alpha       : return D3D11_BLEND_SRC1_ALPHA;
		case Blend::InvSrc1Alpha    : return D3D11_BLEND_INV_SRC1_ALPHA;
			default:
				VCL_NO_SWITCH_DEFAULT;
		}
		
		DebugError("Invalid enum");
		return D3D11_BLEND_ONE;
	}

	D3D11_BLEND_OP D3D::toD3Denum(BlendOp::BlendOp op)
	{
		switch (op)
		{
		case BlendOp::Add         : return D3D11_BLEND_OP_ADD;
		case BlendOp::Subtract    : return D3D11_BLEND_OP_SUBTRACT;
		case BlendOp::RevSubtract : return D3D11_BLEND_OP_REV_SUBTRACT;
		case BlendOp::Min         : return D3D11_BLEND_OP_MIN;
		case BlendOp::Max         : return D3D11_BLEND_OP_MAX;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}
		
		DebugError("Invalid enum");
		return D3D11_BLEND_OP_ADD;
	}

	UINT8 D3D::toD3Denum(ColourWriteEnable::ColourWriteEnable mask)
	{
		UINT8 d3d_mask = 0;
		d3d_mask |= (mask & ColourWriteEnable::Red)   ? D3D11_COLOR_WRITE_ENABLE_RED   : 0;
		d3d_mask |= (mask & ColourWriteEnable::Green) ? D3D11_COLOR_WRITE_ENABLE_GREEN : 0;
		d3d_mask |= (mask & ColourWriteEnable::Blue)  ? D3D11_COLOR_WRITE_ENABLE_BLUE  : 0;
		d3d_mask |= (mask & ColourWriteEnable::Alpha) ? D3D11_COLOR_WRITE_ENABLE_ALPHA : 0;

		return d3d_mask;
	}
	
	D3D11_FILL_MODE D3D::toD3Denum(FillMode::FillMode mode)
	{
		switch (mode)
		{
		case FillMode::Solid     : return D3D11_FILL_SOLID;
		case FillMode::Wireframe : return D3D11_FILL_WIREFRAME;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}
		
		DebugError("Invalid enum");
		return D3D11_FILL_SOLID;
	}

	D3D11_CULL_MODE D3D::toD3Denum(CullMode::CullMode mode)
	{
		switch (mode)
		{
		case CullMode::None  : return D3D11_CULL_NONE;
		case CullMode::Back  : return D3D11_CULL_BACK;
		case CullMode::Front : return D3D11_CULL_FRONT;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}

		DebugError("Invalid enum");
		return D3D11_CULL_NONE;
	}
	
	D3D11_DEPTH_WRITE_MASK D3D::toD3Denum(DepthWriteMask::DepthWriteMask mask)
	{
		switch (mask)
		{
		case DepthWriteMask::Zero : return D3D11_DEPTH_WRITE_MASK_ZERO;
		case DepthWriteMask::All  : return D3D11_DEPTH_WRITE_MASK_ALL;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}

		DebugError("Invalid enum");
		return D3D11_DEPTH_WRITE_MASK_ALL;
	}

	D3D11_COMPARISON_FUNC D3D::toD3Denum(ComparisonFunction::ComparisonFunction func)
	{
		switch (func)
		{
		case ComparisonFunction::Never        : return D3D11_COMPARISON_NEVER;
		case ComparisonFunction::Less         : return D3D11_COMPARISON_LESS;
		case ComparisonFunction::Equal        : return D3D11_COMPARISON_EQUAL;
		case ComparisonFunction::LessEqual    : return D3D11_COMPARISON_LESS_EQUAL;
		case ComparisonFunction::Greater      : return D3D11_COMPARISON_GREATER;
		case ComparisonFunction::NotEqual     : return D3D11_COMPARISON_NOT_EQUAL;
		case ComparisonFunction::GreaterEqual : return D3D11_COMPARISON_GREATER_EQUAL;
		case ComparisonFunction::Always       : return D3D11_COMPARISON_ALWAYS;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}

		DebugError("Invalid enum");
		return D3D11_COMPARISON_LESS;
	}
	
	D3D11_STENCIL_OP D3D::toD3Denum(StencilOperation::StencilOperation func)
	{
		switch (func)
		{
		case StencilOperation::Keep             : return D3D11_STENCIL_OP_KEEP;
		case StencilOperation::Zero             : return D3D11_STENCIL_OP_ZERO;
		case StencilOperation::Replace          : return D3D11_STENCIL_OP_REPLACE;
		case StencilOperation::IncreaseSaturate : return D3D11_STENCIL_OP_INCR_SAT;
		case StencilOperation::DecreaseSaturate : return D3D11_STENCIL_OP_DECR_SAT;
		case StencilOperation::Invert           : return D3D11_STENCIL_OP_INVERT;
		case StencilOperation::Increase         : return D3D11_STENCIL_OP_INCR;
		case StencilOperation::Decrease         : return D3D11_STENCIL_OP_DECR;
		default:
			VCL_NO_SWITCH_DEFAULT;
		}

		DebugError("Invalid enum");
		return D3D11_STENCIL_OP_REPLACE;
	}*/
}}}
