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
#include <iostream>
#include <string>
#include <vector>

// VCL

namespace Vcl { namespace Util
{
	class StringParser
	{
	public:
		StringParser();

	public:
		bool eos() { return _eos; }

	public:
		void setInputStream(std::istream* stream);
		bool loadLine();
		void readLine(std::string* out_string_ptr);
		void skipWhiteSpace();
		void skipLine();
		bool readString(std::string* out_string_ptr);
		bool readFloat(float* f_ptr);
		bool readInt(int* i_ptr);

	private: // Parser state
		//! Stream to parse data from
		std::istream* _stream{ nullptr };

		//! Size of the parse buffer
		static const size_t BufferSize{ 512 * 1024 };

		//! Intermediate parse buffer
		std::vector<char> _streamBuffer;

		//! Start of the current buffer
		char* _currentBuffer{ nullptr };

		//! Size of the current buffer
		size_t _currentSizeAvailable{ 0 };

		//! Current read pointer
		char* _bufferReadPtr{ nullptr };

		//! Reached end of stream?
		bool _eos{ false };
	};
}}
