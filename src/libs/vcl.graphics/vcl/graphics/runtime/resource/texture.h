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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <array>
#include <memory>
#include <vector>

// VCL
#include <vcl/core/flags.h>
#include <vcl/core/optional.h>
#include <vcl/core/span.h>
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace Runtime {
	VCL_DECLARE_FLAGS(TextureUsage,

					  //! Texture can be the source of copy opertions
					  CopySrc,

					  //! Texture can be the destination of copy opertions
					  CopyDst,

					  //! Texture can be used to sample texels from in the shader
					  Sampled,

					  //! Texture can be read as image (e.g. in compute shader)
					  Storage,

					  //! Texture can be used as render-target
					  OutputAttachment)

	enum class TextureCubeFace
	{
		PositiveX = 0,
		NegativeX = 1,
		PositiveY = 2,
		NegativeY = 3,
		PositiveZ = 4,
		NegativeZ = 5
	};

	//! Clear value of a texture. Colour textures use 'Colour', depth-stencil
	//! textures 'Depth' and 'Stencil'.
	struct TextureClearValue
	{
		std::array<float, 4> Colour{ 0, 0, 0, 0 };
		float Depth{ 1 };
		uint8_t Stencil{ 0 };
	};

	struct TextureBaseDescription
	{
		SurfaceFormat Format;
		Flags<TextureUsage> Usage;

		//! Formats, other than 'Format', views on the texture may use.
		//! Reinterpreting the format has to be declared when the texture is
		//! created (Vulkan: mutable format; D3D12: castable formats;
		//! WebGPU: view formats; OpenGL: same format compatibility class).
		std::vector<SurfaceFormat> ViewFormats;

		//! Value the texture is usually cleared to. Clearing a render-target or
		//! depth-stencil texture to this value is fast on D3D12 (optimised clear
		//! value); the other backends ignore it.
		stdext::optional<TextureClearValue> ClearValue;
	};

	struct Texture1DDescription : public TextureBaseDescription
	{
		uint32_t Width;
		uint32_t MipLevels;
		uint32_t ArraySize;
	};

	struct Texture2DDescription : public TextureBaseDescription
	{
		uint32_t Width;
		uint32_t Height;
		uint32_t MipLevels;
		uint32_t ArraySize;
	};

	struct Texture2DMSDescription : public TextureBaseDescription
	{
		uint32_t Width;
		uint32_t Height;
		uint32_t ArraySize;
		uint32_t Samples;
	};

	struct TextureCubeDescription : public TextureBaseDescription
	{
		uint32_t Width;
		uint32_t Height;
		uint32_t MipLevels;
		uint32_t ArraySize;
	};

	struct Texture3DDescription : public TextureBaseDescription
	{
		uint32_t Width;
		uint32_t Height;
		uint32_t Depth;
		uint32_t MipLevels;
	};

	class TextureResource
	{
	public:
		//! Pointer to the image data
		stdext::span<const unsigned char> Data;

		//! Format of the image data
		SurfaceFormat Format{ SurfaceFormat::Unknown };

		//! Width of the image data
		int Width{ 0 };
		//! Height of the image data
		int Height{ 1 };
		//! Number of depth-layers in the image data
		int Depth{ 1 };
		//! Number of array-layers in the image data
		int Layers{ 1 };
		//! Number of mipmaps in the image data
		int MipMaps{ 1 };

		//! Width offset of the image data
		int X{ 0 };
		//! Height offset of the image data
		int Y{ 0 };
		//! Depth offset of the image data
		int Z{ 0 };
		//! Layer offset of the image data
		int Layer{ 0 };
		//! Mipmap offset of the image data
		int MipMap{ 0 };

		const void* data() const
		{
			return reinterpret_cast<const void*>(Data.data());
		}

		int rowPitch() const
		{
			return sizeInBytes(Format) * Width;
		}

		int slicePitch() const
		{
			return Height * rowPitch();
		}

		bool verify() const
		{
			return Data.size() == slicePitch() * Depth * Layers * MipMaps;
		}
	};

	enum class TextureType
	{
		Unknown = 0,
		Texture1D,
		Texture1DArray,
		Texture2D,
		Texture2DArray,
		Texture2DMS,
		Texture2DMSArray,
		Texture3D,
		TextureCube,
		TextureCubeArray
	};

	//! Description of a view on a part of a texture.
	//! The default describes the entire texture.
	struct TextureViewDescription
	{
		//! Value selecting all mip-map levels or layers starting at the first one
		enum : uint32_t
		{
			Remaining = 0xffffffffu
		};

		//! Type the texture is viewed as. 'Unknown' selects the type of the texture.
		TextureType Type{ TextureType::Unknown };

		//! Format the texture is viewed as. 'Unknown' selects the format of the texture.
		SurfaceFormat Format{ SurfaceFormat::Unknown };

		//! First mip-map level, relative to the texture
		uint32_t FirstMipLevel{ 0 };

		//! Number of mip-map levels.
		//! Read-write (storage) views cover exactly one level.
		uint32_t MipLevels{ Remaining };

		//! First array layer, relative to the texture
		uint32_t FirstLayer{ 0 };

		//! Number of array layers
		uint32_t Layers{ Remaining };
	};

	/*!
	 * \brief Texture resource
	 *
	 * Describes the entire resource. Parts of a texture are addressed with a
	 * TextureViewDescription, which each backend turns into its own view object.
	 */
	class Texture
	{
	protected:
		Texture() = default;
		Texture(const Texture&) = default;

		Texture& operator=(const Texture&) = delete;

	public:
		virtual ~Texture() = default;

	public:
		virtual std::unique_ptr<Texture> clone() const = 0;

	public:
		TextureType type() const { return _type; }

		SurfaceFormat format() const { return _format; }

		Flags<TextureUsage> usage() const { return _usage; }

		//! \returns the formats, other than format(), views may use
		stdext::span<const SurfaceFormat> viewFormats() const { return _viewFormats; }

		//! \returns whether a view on the texture may use \p format.
		//!          'Unknown' selects the format of the texture.
		bool supportsViewFormat(SurfaceFormat format) const;

		int width() const { return _width; }
		int height() const { return _height; }
		int depth() const { return _depth; }

		int layers() const { return _nrLayers; }

		int mipMapLevels() const { return _nrLevels; }

		//! \returns the size of a single sub resource of the largest mip-map level
		size_t sizeInBytes() const { return _sizeInBytes; }

	protected:
		void initializeTexture(
			TextureType t,
			const TextureBaseDescription& desc,
			int nrLvls,
			int nrLayers,
			int width,
			int height = 1,
			int depth = 1);

	private:
		//! Texture type
		TextureType _type;

		//! Surface format
		SurfaceFormat _format;

		//! Configured texture usages
		Flags<TextureUsage> _usage;

		//! Formats views may use in addition to the surface format
		std::vector<SurfaceFormat> _viewFormats;

		int _nrLevels;
		int _nrLayers;

	private: // Size of a single sub resource of the largest mip-map level
		int _width;
		int _height;
		int _depth;

	private:
		size_t _sizeInBytes;
	};
}}}
