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
#include <vcl/util/stringparser.h>

// C++ standard library
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace Vcl { namespace Util
{
	StringParser::StringParser()
	: _streamBuffer(BufferSize)
	, _currentBuffer(_streamBuffer.data())
	, _bufferReadPtr(_currentBuffer)
	{
	}

	void StringParser::setInputStream(std::istream* stream)
	{
		_stream = stream;

		// Reset the pointers
		_currentBuffer = _streamBuffer.data();
		_bufferReadPtr = _currentBuffer;
	}

	bool StringParser::loadLine()
	{
		// Check if a full line is still available
		char* current_pointer = _bufferReadPtr;
		while (current_pointer < _currentBuffer + _currentSizeAvailable)
		{
			if ((*current_pointer) == '\n')
			{
				return true;
			}
			++current_pointer;
		}

		// Else, there is no full line available.
		// Copy any remaining data to the beginning and fill up the buffer with new data
		bool full_line_available = false;
		while (!full_line_available)
		{
			// Copy the remaining data to the front
			auto copy_amount = static_cast<size_t>((_currentBuffer + _currentSizeAvailable) - _bufferReadPtr);
			if (copy_amount > 0 && _bufferReadPtr != _currentBuffer)
				// There was some data read, move the remaining
			{
				memmove(_currentBuffer, _bufferReadPtr, copy_amount);
			}
			else if (copy_amount == 0 && _stream->eof())
				// All data was read and we're are at the end of the stream
			{
				_eos = true;
				return false;
			}

			// We want to fill the buffer again
			_bufferReadPtr = _currentBuffer;
			_currentSizeAvailable = copy_amount;

			if (_stream->eof())
			{
				full_line_available = true;
				continue;
			}

			_stream->read(_currentBuffer + _currentSizeAvailable, static_cast<std::streamsize>(BufferSize - _currentSizeAvailable));
			auto amount_read = _stream->gcount();
			if (amount_read == 0)
			{
				throw std::runtime_error("");
			}
			_currentSizeAvailable += static_cast<size_t>(amount_read);

			current_pointer = _bufferReadPtr;
			while (current_pointer < _currentBuffer + _currentSizeAvailable)
			{
				if ((*current_pointer) == '\n')
				{
					full_line_available = true;
					break;
				}
				++current_pointer;
			}
		}

		return full_line_available;
	}

	void StringParser::skipWhiteSpace()
	{
		while (_bufferReadPtr < _currentBuffer + _currentSizeAvailable && (*_bufferReadPtr) <= ' ')
		{
			++_bufferReadPtr;
		}
	}

	void StringParser::skipLine()
	{
		while (_bufferReadPtr < _currentBuffer + _currentSizeAvailable && (*_bufferReadPtr) != '\n')
		{
			++_bufferReadPtr;
		}
		if (_bufferReadPtr < _currentBuffer + _currentSizeAvailable)
		{
			++_bufferReadPtr;
		}
	}

	void StringParser::readLine(std::string* out_string_ptr)
	{
		char* begin_ptr = _bufferReadPtr;
		skipLine();
		
		// Store the end-pointer and terminate the string
		char c = *_bufferReadPtr;
		*_bufferReadPtr = '\0';

		// Store the string
		*out_string_ptr = begin_ptr;

		// Restore the stored end-pointer
		*_bufferReadPtr = c;
	}

	bool StringParser::readString(std::string* out_string_ptr)
	{
		skipWhiteSpace();

		char* begin_ptr = _bufferReadPtr;
		while ((*_bufferReadPtr) > ' ')
		{
			++_bufferReadPtr;
		}

		if (_bufferReadPtr == begin_ptr)
		{
			return false;
		}
		
		// Store the end-pointer and terminate the string
		char c = *_bufferReadPtr;
		*_bufferReadPtr = '\0';

		// Store the string
		*out_string_ptr = begin_ptr;

		// Restore the stored end-pointer
		*_bufferReadPtr = c;

		return true;
	}

	bool StringParser::readFloat(float* f_ptr)
	{
		skipWhiteSpace();

		// Find the end of the number
		char* begin_ptr = _bufferReadPtr;
		while
		(
			((*_bufferReadPtr) >= '0' && (*_bufferReadPtr) <= '9') ||
			 (*_bufferReadPtr) == '.' ||
			 (*_bufferReadPtr) == '-' ||
			 (*_bufferReadPtr) == 'E' ||
			 (*_bufferReadPtr) == 'e'
		)
		{
			++_bufferReadPtr;
		}

		if (_bufferReadPtr == begin_ptr)
		{
			return false;
		}

		// Store the end-pointer and terminate the string
		char c = *_bufferReadPtr;
		*_bufferReadPtr = '\0';

		// Convert the string to a float
		*f_ptr = static_cast<float>(atof(begin_ptr));

		// Restore the stored end-pointer
		*_bufferReadPtr = c;

		return true;
	}

	bool StringParser::readInt(int* i_ptr)
	{
		skipWhiteSpace();

		// Find the end of the number
		char* begin_ptr = _bufferReadPtr;
		while (((*_bufferReadPtr) >= '0' && (*_bufferReadPtr) <= '9') ||
		        (*_bufferReadPtr) == '-')
		{
			++_bufferReadPtr;
		}

		if (_bufferReadPtr == begin_ptr)
		{
			return false;
		}

		// Store the end-pointer and terminate the string
		char c = *_bufferReadPtr;
		*_bufferReadPtr = '\0';

		// Convert the string to an int
		*i_ptr = atoi(begin_ptr);

		// Restore the stored end-pointer
		*_bufferReadPtr = c;

		return true;
	}
}}
