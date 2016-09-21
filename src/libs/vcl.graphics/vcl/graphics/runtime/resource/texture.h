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

// VCL
#include <vcl/graphics/surfaceformat.h>

namespace Vcl { namespace Graphics { namespace Runtime
{
	enum class TextureCubeFace
	{
		PositiveX = 0,
		NegativeX = 1,
		PositiveY = 2,
		NegativeY = 3,
		PositiveZ = 4,
		NegativeZ = 5
	};

	struct TextureBaseDescription
	{
		SurfaceFormat Format;
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
		void* Data;

		//! Format of the image data
		SurfaceFormat Format;

		//! Width of the image data
		int Width;
		int Height;
		int Depth;
		int Layers;
		int MipMaps;

	public:
		TextureResource()
		: Data(nullptr)
		, Format(SurfaceFormat::Unknown)
		, Width(1)
		, Height(1)
		, Depth(1)
		, Layers(1)
		, MipMaps(1)
		{
		}

		int rowPitch() const
		{
			return sizeInBytes(Format) * Width;
		}

		int slicePitch() const
		{
			return Height * rowPitch();
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

	class TextureView
	{
	protected:
		TextureView() = default;
		TextureView(const TextureView&) = default;

		TextureView& operator= (const TextureView&) = delete;

	public:
		virtual ~TextureView() = default;


	public:
		TextureType type() const { return _type; }

		SurfaceFormat format() const { return _format; }

		int width() const { return _width; }
		int height() const { return _height; }
		int depth() const { return _depth; }

		int firstLayer() const { return _layer; }
		int layers() const { return _nrLayers; }

		int firstMipMapLevel() const { return _level; }
		int mipMapLevels() const { return _nrLevels; }

		size_t sizeInBytes() const { return _sizeInBytes; }

	protected:
		void initializeView
		(
			TextureType t, SurfaceFormat f,
			int firstLvl, int nrLvls,
			int firstLayer, int nrLayers,
			int width, int height = 1, int depth = 1
		);

	private:
		//! Texture type
		TextureType _type;

		//! Surface format
		SurfaceFormat _format;

		int  _level;
		int  _nrLevels;
		int  _layer;
		int  _nrLayers;

	private: // Size of a single sub resource of the lowest mip-map level
		int _width;
		int _height;
		int _depth;

	private:
		size_t _sizeInBytes;
	};

	class Texture : public TextureView
	{
	protected:
		Texture() = default;
		Texture(const Texture&) = default;

	public:
		virtual ~Texture() = default;

	public:
		virtual std::unique_ptr<Texture> clone() const = 0;

	public:
		virtual void clear(SurfaceFormat fmt, const void* data) = 0;
	};
}}}
