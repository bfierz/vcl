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

// VCL library
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace Vcl { namespace IO {
	class Bitmap
	{
	public:
		static void store(const std::string& filename, int width, int height, const std::vector<std::array<unsigned char, 4>>& data)
		{
			using namespace std;

			// Bmp Header
			struct Header
			{
				short type;
				int size;
				short reserved1,
					reserved2;
				int offset;
			};

			struct Info
			{
				int Size;
				long Width;
				long Height;
				short Planes;
				short BitCount;
				int Compression;
				int SizeImage;
				long XPelsPerMeter;
				long YPelsPerMeter;
				int ClrUsed;
				int ClrImportant;
			};

			struct Colour
			{
				unsigned char b;
				unsigned char g;
				unsigned char r;
				unsigned char a;
			};

			// Create image
			Header header = { 19778, static_cast<int>(sizeof(Colour) * width * height), 0, 0, 54 };
			Info info = { sizeof(Info), width, height, 1, sizeof(Colour) * 8, 0, static_cast<int>(sizeof(Colour) * width * height), 1, 1, 0, 0 };

			fstream img;
			img.open(filename, ios::out | ios::binary);
			if (!img.is_open())
				return;

			// Write header
			img.write(reinterpret_cast<char*>(&header.type), sizeof(short));
			img.write(reinterpret_cast<char*>(&header.size), sizeof(int));
			img.write(reinterpret_cast<char*>(&header.reserved1), sizeof(short));
			img.write(reinterpret_cast<char*>(&header.reserved2), sizeof(short));
			img.write(reinterpret_cast<char*>(&header.offset), sizeof(int));

			img.write(reinterpret_cast<char*>(&info), sizeof(info));

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					const std::array<unsigned char, 4>& c = data[i * width + j];
					Colour pixel;
					pixel.a = c[3];
					pixel.b = c[2];
					pixel.g = c[1];
					pixel.r = c[0];

					img.write(reinterpret_cast<char*>(&pixel), sizeof(Colour));
				}
			}
			img.close();
		}
	};
}}
